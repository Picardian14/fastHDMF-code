/**
 * C mex interface to fast DMF simulator to generate BOLD signals. See
 * README.md and DMF.hpp for details.
 *
 * Pedro Mediano, Apr 2020
 */

#include "mex.h"
#include "./dyn_fic_DMF.hpp"
#include<string>

/**
 * Cross-platform way of getting the pointer to the raw data in a mxArray. Uses
 * preprocessor directives to switch between pre- and post-2017 mex functions.
 *
 * NOTE: assumes that the array has doubles, but does not explicitly check.
 *
 */
double* safeGet(const mxArray* a) {
    double *p;
    #if MX_HAS_INTERLEAVED_COMPLEX
    p = mxGetDoubles(a);
    #else
    p = mxGetPr(a);
    #endif
    return p;
}


/**
 * Converts a Matlab struct with double fields into a std::map, keeping
 * pointers to all arrays without making any deep copies.
 *
 * In addition, for each field `f` adds a field named `isvector_$f` which
 * contains a null pointer if `f` points to a 1x1 Matlab scalar, or a non-null
 * pointer otherwise.
 *
 * @param structure_array_ptr pointer to Matlab-like struct
 * @return ParamStruct containing pointers to the Matlab array data
 */
ParamStruct struct2map(const mxArray *structure_array_ptr) {
    mwSize total_num_of_elements;
    int number_of_fields, field_index;
    const char  *field_name;
    const mxArray *field_array_ptr;
    ParamStruct p;
    double *x;

    total_num_of_elements = mxGetNumberOfElements(structure_array_ptr);
    number_of_fields = mxGetNumberOfFields(structure_array_ptr);

    for (field_index=0; field_index<number_of_fields; field_index++)  {
        field_name = mxGetFieldNameByNumber(structure_array_ptr, field_index);
        field_array_ptr = mxGetFieldByNumber(structure_array_ptr,
                                             0, field_index);
        x = safeGet(field_array_ptr);
        p[field_name] = x;

        std::string buffer("isvector_");
        buffer.append(field_name);
        bool isvector = mxGetN(field_array_ptr) > 1 || mxGetM(field_array_ptr) > 1;
        p[buffer] = isvector ? x : NULL;
    }

    try {
      // Use checkParams to validate and add backward compatibility
      checkParams(p);
    } catch (const std::invalid_argument& e) {
      mexErrMsgIdAndTxt("DMF:invalidArgument", e.what());
    }

    return p;
}


/**
 * Check that input and output arguments to MEX function are valid.
 *
 * Makes sure that 1) number and type of inputs is correct, 2) struct has
 * all necessary fields, and 3) number of outputs is correct. If arguments
 * are not valid, throws a Matlab error.
 *
 * @throws Matlab error if input or output arguments are invalid
 */
void checkArguments(int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[]) {
    // Check that number and type of inputs is correct
    if(nrhs < 2 || nrhs > 3) {
        mexErrMsgIdAndTxt("DMF:nrhs","Two or three inputs required.");
    }

    /* make sure the first input argument is type struct */
    if (!mxIsStruct(prhs[0])) {
        mexErrMsgIdAndTxt("DMF:notStruct","First input must be a struct.");
    }
    
    /* make sure the second input argument is type double */
    if(!mxIsDouble(prhs[1])) {
        mexErrMsgIdAndTxt("DMF:notDouble","Second input must be type double.");
    }

    // Make sure there are no empty arrays in struct
    int nfields = mxGetNumberOfFields(prhs[0]);
    for (int i = 0; i < nfields; i++) {
      auto f = mxGetFieldByNumber(prhs[0], 0, i);
      if (mxIsEmpty(f)) {
        mexErrMsgIdAndTxt("DMF:badInputs", "Empty arrays not allowed in input struct.");
      }
    }
    
    ParamStruct params = struct2map(prhs[0]);
    
    // Calculate expected number of outputs based on return flags
    int return_flags = DYN_FIC_DMFSimulator::calculateReturnFlags(params);
    int expected_outputs = 0;
    
    if (return_flags & RETURN_RATE_E) expected_outputs++;
    if (return_flags & RETURN_RATE_I) expected_outputs++;
    if (return_flags & RETURN_BOLD) expected_outputs++;
    if (return_flags & RETURN_FIC) expected_outputs++;
    
    if (nlhs != expected_outputs) {
        mexErrMsgIdAndTxt("DMF:badOutputs", "Wrong number of output arguments.");
    }
}


/**
 * Main function to call DMF from Matlab, using the C Mex interface.
 */
void mexFunction(int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[]) {
    checkArguments(nlhs, plhs, nrhs, prhs);

    // First input argument: parameter struct
    ParamStruct params = struct2map(prhs[0]);
    size_t N = (size_t) mxGetN(mxGetField(prhs[0], 0, "C"));

    // Second input argument: number of steps
    size_t nb_steps = (size_t) safeGet(prhs[1])[0];

    // Get return flags and plasticity mode
    int return_flags = DYN_FIC_DMFSimulator::calculateReturnFlags(params);
    PlasticityMode plasticity_mode = static_cast<PlasticityMode>(static_cast<int>(params["plasticity_mode"][0]));
    
    // Pre-allocate memory for results using Matlab's factory
    size_t nb_steps_bold = nb_steps * params["dtt"][0] / params["TR"][0];
    size_t batch_size = params["batch_size"][0];

    // Create arrays only as needed, with appropriate sizes
    mxArray *rate_e_res = NULL, *rate_i_res = NULL, *bold_res = NULL, *fic_res = NULL;
    
    // Always allocate minimum arrays needed for computation
    if (return_flags & RETURN_RATE_E) {
        rate_e_res = mxCreateDoubleMatrix(N, nb_steps, mxREAL);
    } else {
        // Need minimal buffer for internal calculations
        rate_e_res = mxCreateDoubleMatrix(N, 2*batch_size, mxREAL);
    }
    
    if (return_flags & RETURN_RATE_I) {
        rate_i_res = mxCreateDoubleMatrix(N, nb_steps, mxREAL);
    } else {
        // Need minimal buffer for internal calculations
        rate_i_res = mxCreateDoubleMatrix(N, 2*batch_size, mxREAL);
    }
    
    if (return_flags & RETURN_FIC) {
        fic_res = mxCreateDoubleMatrix(N, nb_steps, mxREAL);
    } else {
        // Minimal buffer if needed internally
        fic_res = mxCreateDoubleMatrix(N, 2*batch_size, mxREAL);
    }
    
    if (return_flags & RETURN_BOLD) {
        bold_res = mxCreateDoubleMatrix(N, nb_steps_bold, mxREAL);
    } else {
        // Don't need BOLD buffer if not returning
        bold_res = mxCreateDoubleMatrix(N, 1, mxREAL);
    }

    // Run, passing results by reference
    try {      
        DYN_FIC_DMFSimulator sim(params, nb_steps, N);
        sim.run(safeGet(rate_e_res), safeGet(rate_i_res), safeGet(bold_res), safeGet(fic_res));
      
    } catch (...) {
        mexErrMsgIdAndTxt("DMF:unknown", "Unknown failure occurred.");
    }

    // Copy back results with cleaner bitflag-based output assignment
    int output_idx = 0;
    
    // Return requested outputs in order (E-rates, I-rates, BOLD, FIC)
    if (return_flags & RETURN_RATE_E) {
        plhs[output_idx++] = rate_e_res;
    }
    
    if (return_flags & RETURN_RATE_I) {
        plhs[output_idx++] = rate_i_res;
    }
    
    if (return_flags & RETURN_BOLD) {
        plhs[output_idx++] = bold_res;
    }
    
    if (return_flags & RETURN_FIC) {
        plhs[output_idx++] = fic_res;
    }
}

