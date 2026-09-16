/**
 * Optimised C++ implementation of the Dynamic Mean Field (DMF) model with a
 * BOLD Balloon-Windkessel model.
 *
 * Pedro Mediano, Apr 2020
 */

#ifndef DMF_H
#define DMF_H

#include<string>
#include<thread>
#include<iostream>
#include<random>
#include<map>
using namespace std;
#include "Eigen/Dense"

typedef std::map<std::string, const double*> ParamStruct;

/**
 * Simple utility struct to pass ranges of integer values.
 *
 * Example usage:
 *
 * int start=0, end=0;
 * Range r(start, end);
 *
 */
struct Range {
    size_t start, end;
    Range(size_t s, size_t e) : start(s), end(e) {}
};


/**
 * Clip value of double between 0 and 1.
 */
inline double clip(double x) {
    return x > 1 ? 1 : (x < 0 ? 0 : x);
}


/**
 * Retrieves a field from a ParamStruct and makes sure that result is an array
 * of size N. If the content of the field is a scalar, it constructs a constant
 * array.
 *
 * @param params
 * @param field name of field to be retrieved
 * @param N expected size of the array
 */
Eigen::ArrayXd ensureArray(const ParamStruct &params, std::string field, size_t N) {

    // Initialise array and field name string
    Eigen::ArrayXd A;
    std::string isvecf("isvector_");
    isvecf.append(field);

    if (params.at(isvecf.c_str())) {
        // Map the Eigen::Array to field if it is listed as a vector
        A = Eigen::Map<const Eigen::ArrayXd>(params.at(field.c_str()), N);

    } else {
        // Otherwise initialise and fill a constant array
        A = Eigen::ArrayXd::Zero(N);
        A.fill(params.at(field.c_str())[0]);
    }

    return A;
}


/**
 * Checks input parameter structure has all necessary fields.
 *
 * @param params
 */
void checkParams(const ParamStruct &params) {

    // Check that parameter struct has the necessary fields
    std::vector<std::string> required_fields = {"C", "receptors", "dt",
            "taon", "taog", "gamma", "sigma", "JN", "I0", "Jexte", "Jexti",
            "w", "g_e", "Ie", "ce", "g_i", "Ii", "ci", "wgaine", "wgaini",
            "G", "TR", "dtt", "batch_size",
            "return_rate", "return_bold","return_fic", 
            "lrj","taoj","with_plasticity", "with_decay","obj_rate",
            "nvc_sigmoid", "nvc_match_slope", "nvc_r0", "nvc_u50",
        }; // added ljr, taoj and obj_rate as parameters of fic dynamics
    for (const auto& field : required_fields) {
        if (!params.count(field.c_str())) {
            std::string s("Missing field in parameter struct: ");
            s.append(field);
            throw std::invalid_argument(s);
        }
    }
}


/**
 * Integrator of firing rates to simulate BOLD signals
 */
class BOLDIntegrator {


public:
    // Balloon-Windkessel equation parameters
    double taus  = 0.65;
    double tauf  = 0.41;
    double tauo  = 0.98;
    double alpha = 0.32;

    double Eo = 0.4;
    double TE = 0.04;
    double vo = 0.04;
    double k1 = 4.3*40.3*Eo*TE;
    double k2 = 25*Eo*TE;
    double k3 = 1;

    double itaus  = 1/taus;
    double itauf  = 1/tauf;
    double itauo  = 1/tauo;
    double ialpha = 1/alpha;
    double dt;

    // --- Neurovascular coupling (NVC) controls (inside integrator) ---
    bool   nvc_sigmoid     = false;  // default: off -> original behavior
    bool   nvc_match_slope = true;   // match small-signal gain at baseline
    double nvc_r0          = 3.4;    // baseline firing-rate (Hz). If your r is already baseline-subtracted, set 0.0
    double nvc_u50         = 15.0;   // half-saturation (Hz): compression starts in 15–30 Hz range
    double nvc_umax        = 1.0;    // logistic asymptote (arbitrary units)
    double nvc_k           = 0.25;   // logistic steepness (1/Hz). Larger -> sharper saturation


    size_t N, save_every, nb_bold_steps, rate_buffer_size;
    size_t count = 1;  // Start in 1 to match subsampling in Deco code
    size_t b_idx = 0;
    size_t i = 0;
    std::thread th;

    Eigen::Map<Eigen::MatrixXd> b;
    Eigen::Map<Eigen::ArrayXXd> r;    
    Eigen::Map<Eigen::ArrayXd> s, f, v, q;
    Eigen::Map<Eigen::ArrayXd> ds, df, dv, dq;
    Eigen::ArrayXd Z;
    Eigen::ArrayXd dZ;

    /**
     * Construct integrator.
     *
     * @param params Matlab struct with all necessary parameters (see checkArguments)
     * @param nb_rate_steps
     * @param N_in
     */
    BOLDIntegrator(ParamStruct params, size_t nb_rate_steps, size_t N_in) :
              dt(params["dtt"][0]),
              N(N_in),
              save_every(params["TR"][0]/params["dtt"][0]),
              nb_bold_steps(nb_rate_steps*params["dtt"][0]/params["TR"][0]),
              rate_buffer_size(nb_rate_steps),
              nvc_sigmoid(params["nvc_sigmoid"][0]),
              nvc_match_slope(params["nvc_match_slope"][0]),
              nvc_r0(params["nvc_r0"][0]),
              nvc_u50(params["nvc_u50"][0]),
              b(NULL, N_in, nb_rate_steps*params["dtt"][0]/params["TR"][0]),
              r(NULL, N_in, nb_rate_steps),              
              s(NULL, N_in),
              f(NULL, N_in),
              v(NULL, N_in),
              q(NULL, N_in),
              ds(NULL, N_in),
              df(NULL, N_in),
              dv(NULL, N_in),
              dq(NULL, N_in),
              Z(Eigen::ArrayXd::Ones(4*N_in)),
              dZ(4*N_in)
    {}

    /**
     * Initialise maps and arrays.
     *
     * Only two arrays are allocated, Z (value) and dZ (derivative) with the
     * state of the BW model. For readability, chunks of these arrays are split
     * and mapped to the standard variables in the BW model (s,f,v,q).
     *
     * As the integrator runs, fills a pre-allocated array with BOLD values.
     *
     * @param[in] rate_e_res reference to firing rate array
     * @param[out] bold_res writeable reference to BOLD array
     */
    void init(double* rate_e_res, double* bold_res) {
    //void init(double* rate_e_res, double* bold_res, double* fic_res) {

        double *Z_ptr = &Z(0), *dZ_ptr = &dZ(0);
        new (&s) Eigen::Map<Eigen::ArrayXd>(Z_ptr, N);
        new (&f) Eigen::Map<Eigen::ArrayXd>(Z_ptr + 1*N, N);
        new (&v) Eigen::Map<Eigen::ArrayXd>(Z_ptr + 2*N, N);
        new (&q) Eigen::Map<Eigen::ArrayXd>(Z_ptr + 3*N, N);
        new (&ds) Eigen::Map<Eigen::ArrayXd>(dZ_ptr, N);
        new (&df) Eigen::Map<Eigen::ArrayXd>(dZ_ptr + 1*N, N);
        new (&dv) Eigen::Map<Eigen::ArrayXd>(dZ_ptr + 2*N, N);
        new (&dq) Eigen::Map<Eigen::ArrayXd>(dZ_ptr + 3*N, N);
        
        s.fill(0);

        new (&b) Eigen::Map<Eigen::MatrixXd>(bold_res, N, nb_bold_steps);
        new (&r) Eigen::Map<Eigen::ArrayXXd>(rate_e_res, N, rate_buffer_size);        

    }

    /**
     * Compute BW equations for the value of firing rates at position idx in
     * the shared array.
     *
     * @param idx index of firing rate to use
     */
    void compute(size_t idx) {
        // --- BEGIN: NVC (sigmoidal saturation) ---------------------------------------
    Eigen::ArrayXd z_arr;  // elementwise drive (array for .exp(), etc.)

    const Eigen::ArrayXd u = r.col(idx % rate_buffer_size).array();

    // Build a baseline array (same size as u)
    const Eigen::ArrayXd u0 = Eigen::ArrayXd::Constant(u.size(), nvc_r0);

    // Logistic g(u) = umax / (1 + exp(-k * (u - u50)))
    auto g_of = [&](const Eigen::ArrayXd& x) -> Eigen::ArrayXd {
        return nvc_umax / (1.0 + (-nvc_k * (x - nvc_u50)).exp());
    };

    // g'(u) = umax * k * exp(-k(u-u50)) / (1 + exp(-k(u-u50)))^2
    auto gp_of = [&](const Eigen::ArrayXd& x) -> Eigen::ArrayXd {
        const Eigen::ArrayXd e   = (-nvc_k * (x - nvc_u50)).exp();
        const Eigen::ArrayXd den = 1.0 + e;
        return nvc_umax * nvc_k * e / (den * den);
    };
    if (nvc_sigmoid) {
        // u: the neural drive you were using (current column of rates buffer)

        const Eigen::ArrayXd g_u  = g_of(u);
        const Eigen::ArrayXd g_u0 = g_of(u0);

        // Slope-match so small deviations around baseline behave like original mapping
        if (nvc_match_slope) {
            Eigen::ArrayXd gp0 = gp_of(u0);
            // Guard against tiny slopes to avoid blow-ups
            const double eps = 1e-12;
            gp0 = gp0.max(eps);
            z_arr = (g_u - g_u0) / gp0;  // elementwise scaling
        } else {
            z_arr = (g_u - g_u0);
        }
    } else {
        // Original behavior’s drive (no saturation). If your original code used r.col(...),
        // keep it here so the only change below is the replacement of that symbol by 'z'.
        z_arr = u;
    }    
    // --- END: NVC -----------------------------------------------------------------

    ds = ( z_arr - itaus*s - itauf*(f - 1) );
    df = s;
    dv = itauo*(f - v.pow(ialpha));      
    dq = itauo*(f*(1-pow(1-Eo, 1/f))/Eo - (v.pow(ialpha-1))*q);
    
    Z += dt*dZ;
    count++;

    if (count == save_every) {
        b.col(b_idx)  = vo*( k1*(1-q) + k2*(1-q/v) + k3*(1-v) );
        count -= save_every;
        b_idx++;
    }
    }

    /**
     * Compute a full batch of BOLD steps for firing rate values in the given
     * range of the shared array.
     *
     * @param range range of firing rate indices to use
     */
    void compute_range(Range range) {
        for (auto i = range.start; i <= range.end; i++) {
            compute(i);
        }
    }

    /**
     * Spawn a new thread and compute a batch of BOLD steps for firing rate
     * values in the given range of the shared array.
     *
     * @param range range of firing rate indices to use
     */
    void compute_async(Range r) {
        th = std::thread([=] { compute_range(r); });
    }

    /**
     * Join the thread in which the integrator is computing asynchronously,
     * if possible.
     *
     * Simple wrapper around std::thread::join.
     */
    void join() {
        if (th.joinable()) { th.join(); }
    }

    /**
     * Make sure that copy constructors are disabled (to avoid making copies
     * of this object when passing it to threads).
     */
    BOLDIntegrator(const BOLDIntegrator&) = delete;
    void operator=(const BOLDIntegrator&) = delete;

};


/**
 * Main class of the DMF simulator.
 *
 * Integrates the system of ODEs of the Dynamic Mean Field model using a
 * stochastic Euler-Maruyama method. Includes a parallelised integrator for the
 * BOLD Balloon-Windkessel model to simulate fMRI time series.
 *
 * RH Sep. 2021. Includes Plasticity rule for the local inihibitory feedback
 *
 */
class DYN_FIC_DMFSimulator {

public:
    double dt;
    double I0;
    double w;
    double JN;
    Eigen::MatrixXd C;
    double G;
    double gamma;
    double sigma;
    double taog;
    double taon;
    double wgaine;
    double wgaini;
    double g_e;
    double g_i;
    double Ie;
    double Ii;
    double ce;
    double ci;
    double dtt;
    double obj_rate; // fic objective rate
    
    

    
    

    size_t nb_steps, N, batch_size, steps_per_millisec, seed;
    bool return_rate, return_bold, return_fic, with_decay, with_plasticity;

    Eigen::ArrayXd sn, sg, J, receptors,lrj,taoj, Jexte, Jexti;

    BOLDIntegrator bold_int;

    /**
     * Constructor.
     *
     * @param params Matlab struct with all necessary parameters (see checkArguments)
     * @param nb_steps_in number of firing rate steps to simulate
     * @param N_in number of nodes/ROIs in the model

     */
    DYN_FIC_DMFSimulator(ParamStruct params, size_t nb_steps_in, size_t N_in) :
            dt(params["dt"][0]),
            I0(params["I0"][0]),
            w(params["w"][0]),
            JN(params["JN"][0]),
            G(params["G"][0]),
            gamma(params["gamma"][0]),
            sigma(params["sigma"][0]),
            taog(params["taog"][0]),
            taon(params["taon"][0]),
            wgaine(params["wgaine"][0]),
            wgaini(params["wgaini"][0]),
            g_e(params["g_e"][0]),
            g_i(params["g_i"][0]),
            Ie(params["Ie"][0]),
            Ii(params["Ii"][0]),
            ce(params["ce"][0]),
            ci(params["ci"][0]),
            dtt(params["dtt"][0]),
            obj_rate(params["obj_rate"][0]),
            nb_steps(nb_steps_in),
            N(N_in),
            batch_size(params["batch_size"][0]),
            steps_per_millisec(1.0/params["dt"][0]),
            return_rate(params["return_rate"][0]),
            return_bold(params["return_bold"][0]),
            return_fic(params["return_fic"][0]),
            with_decay(params["with_decay"][0]),
            with_plasticity(params["with_plasticity"][0]),
            sn(N_in),
            sg(N_in),                        
            bold_int(params, nb_steps, N_in) {

              C = Eigen::Map<const Eigen::MatrixXd>(params["C"], N, N);

              receptors = ensureArray(params, "receptors", N);
              lrj = ensureArray(params, "lrj", N);
              taoj = ensureArray(params, "taoj", N);
              Jexte     = ensureArray(params, "Jexte", N);
              Jexti     = ensureArray(params, "Jexti", N);
              J         = ensureArray(params, "J", N);

              if (params.count("seed")) {
                seed = params["seed"][0];
              } else {
                seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
              }

              Eigen::initParallel();

    };


    inline Eigen::ArrayXd curr2rate(const Eigen::ArrayXd& x, double wgain, double g,
           double I, double c) {
        Eigen::ArrayXd y = c*(x-I)*(1+receptors*wgain);
        return y/(1-exp(-g*y));
    }


    void run(double* rate_e_res,double* rate_i_res, double* bold_res, double* fic_res) {        
        // Initialise BOLD integrator if needed
        if (return_bold) { bold_int.init(rate_e_res, bold_res); }        
        double bold_timer = 0;
        size_t last_bold = 0;

        // Build Eigen::Map to return excitatory rates by reference
        size_t rate_size = return_rate ? nb_steps : (2*batch_size);
        Eigen::Map<Eigen::ArrayXXd> rn(rate_e_res, N, rate_size);

        // Build Eigen::Map to return inhibitory rates by reference        
        Eigen::Map<Eigen::ArrayXXd> rg(rate_i_res, N, rate_size);

        // Build Eigen::Map to return fic by reference
        size_t fic_size = return_fic ? nb_steps : (2*batch_size);
        Eigen::Map<Eigen::ArrayXXd> fic(fic_res, N, fic_size);

        // Initialise PRNG and arrays, and start simulation
        std::default_random_engine e(seed);
        std::normal_distribution<double> n(0, std::sqrt(dt)*sigma);
        sn.fill(0.001);
        sg.fill(0.001);
        Eigen::ArrayXd rnd = Eigen::ArrayXd::Zero(N);
        Eigen::ArrayXd jt = J; //initializing fic
        
        for (size_t t = 0; t < nb_steps; t++) {

            size_t rate_idx = t % rate_size;
            size_t fic_idx = t % fic_size;
            for (size_t dummy = 0; dummy < steps_per_millisec; dummy++) {                
                Eigen::ArrayXd xn = with_plasticity ? I0*Jexte + w*JN*sn + G*JN*(C*sn.matrix()).array() - jt*sg : I0*Jexte + w*JN*sn + G*JN*(C*sn.matrix()).array() - J*sg;;
                Eigen::ArrayXd xg = I0*Jexti + JN*sn - sg;

                rn.col(rate_idx) = curr2rate(xn, wgaine, g_e, Ie, ce);
                rg.col(rate_idx) = curr2rate(xg, wgaini, g_i, Ii, ci);

                rnd = rnd.unaryExpr([&n, &e](double dummy){return n(e);});
                sn += dt*(-sn/taon+(1-sn)*gamma*rn.col(rate_idx)/1000) + rnd; 
                sn = sn.unaryExpr(&clip);

                rnd = rnd.unaryExpr([&n, &e](double dummy){return n(e);});
                sg += dt*(-sg/taog+rg.col(rate_idx)/1000) + rnd;
                sg = sg.unaryExpr(&clip);
                if (with_decay) {
                    jt += dt*(-jt/taoj + lrj*(rg.col(rate_idx)*(rn.col(rate_idx)-obj_rate))/1000000); // plasticity and decay               
                }else{
                    jt += dt*(lrj*(rg.col(rate_idx)*(rn.col(rate_idx)-obj_rate))/1000000); // plasticity
                };                
                
                fic.col(fic_idx) = jt; // saving
            }

            auto start = std::chrono::steady_clock::now();
            if (return_bold && ( ((t+1) % batch_size) == 0)) {
                bold_int.join();

                bold_int.compute_async(Range(rate_idx - batch_size + 1, rate_idx));

                last_bold = (rate_idx + 1) % batch_size;

                #ifdef NO_PARALLEL
                bold_int.join();
                #endif
            }
            auto end = std::chrono::steady_clock::now();
            auto diff = end - start;
            bold_timer += std::chrono::duration <double, std::milli> (diff).count();

        }
        
        if (return_bold) {
            bold_int.join();

            if  ((nb_steps%batch_size) > last_bold) {
                // Compute the remainder of BOLD samples
                // TODO: this will compute things even if they do not end up leading to
                // a new actual BOLD sample. Fix to avoid unnecessary computation.
                bold_int.compute_range(Range(last_bold, nb_steps%batch_size));
            }

          // std::cout << "Bold time: " << bold_timer << std::endl;
        }

    }

};

#endif

