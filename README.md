### Just messing around with the script, despite having no tensorflow experience. What could go wrong?

#### Ideas or whatever:
- Easiest way to debloat the huge script is to put **weights64** in a separate file and `@require` it.

- [Saving and loading models](https://www.tensorflow.org/js/guide/save_load) aka remove the whole iohander and replace:\
  `await tf.loadLayersModel(iohander);`\
  with\
  `await tf.loadLayersModel('https://github.com/Huhni/4chan-captcha-solver/raw/main/model.json');`
  
  This should then load the `./weights.bin` and the whole base64 decoding and **modelJSON** bloat would be gone
  
  --> **Problem:** CORS exists.

- Loading the raw binary weights as a resource\
  `// @resource    weights https://github.com/Huhni/4chan-captcha-solver/raw/main/weights.bin`\
  `var binary_string = window.atob(base64);`  --> `var binary_string = GM_getResourceText('weights')`
  
  --> *Error: Based on the provided shape, [1200,800], the tensor should have 960000 values but has 0*\
  Maybe **GM_getResourceText** can't handle binary data? idk. Perhaps **GM_xmlhttpRequest** with `responseType = 'arraybuffer'` works.
 
 
 ### Other stuff
 - Switch to [tfjs-core](https://www.jsdelivr.com/package/npm/@tensorflow/tfjs-core) and use [WASM as backend](https://www.jsdelivr.com/package/npm/@tensorflow/tfjs-backend-wasm)
 - use minified scripts, for some placebo speed improvement
 - Using [subresource Integrity](https://www.tampermonkey.net/documentation.php#Subresource_Integrity) is also always a good idea
