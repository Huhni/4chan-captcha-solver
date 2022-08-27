// ==UserScript==
// @name        4chan captcha solver
// @namespace   AUTOMATIC
// @match       https://boards.4channel.org/*
// @match       https://boards.4chan.org/*
// @grant       none
// @version     1.1
// @author      AUTOMATIC
// @require     https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.1/dist/tf.min.js#sha256=f33d8906268c993e5a276c88fd8855bf229a2c5d57fe538f2748a3db6e46af96
// @require     https://github.com/Huhni/4chan-captcha-solver/raw/main/weights64.js#sha256=bc28aab77b077565d879e8cc59d502a3f2a20347c8751191f6fbdf7ba0754c3f
// @description modified: model in separate file, fix some linter warnings
// ==/UserScript==


/* global tf, weights64 */

var modelJSON = {"format": "layers-model", "generatedBy": "keras v2.4.0", "convertedBy": "TensorFlow.js Converter v3.7.0", "modelTopology": {"keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, null, 80, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": [null, null, 80, 1], "dtype": "float32", "filters": 40, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "same", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 60, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "same", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": [-1, 1200]}}, {"class_name": "Bidirectional", "config": {"name": "bidi", "trainable": true, "dtype": "float32", "layer": {"class_name": "LSTM", "config": {"name": "lstm", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, "merge_mode": "concat"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 22, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.001, "decay": 0.0, "rho": 0.9, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}, "weightsManifest": [{"paths": ["group1-shard1of1.bin"], "weights": [{"name": "bidi/forward_lstm/lstm_cell_4/kernel", "shape": [1200, 800], "dtype": "float32"}, {"name": "bidi/forward_lstm/lstm_cell_4/recurrent_kernel", "shape": [200, 800], "dtype": "float32"}, {"name": "bidi/forward_lstm/lstm_cell_4/bias", "shape": [800], "dtype": "float32"}, {"name": "bidi/backward_lstm/lstm_cell_5/kernel", "shape": [1200, 800], "dtype": "float32"}, {"name": "bidi/backward_lstm/lstm_cell_5/recurrent_kernel", "shape": [200, 800], "dtype": "float32"}, {"name": "bidi/backward_lstm/lstm_cell_5/bias", "shape": [800], "dtype": "float32"}, {"name": "conv2d/kernel", "shape": [3, 3, 1, 40], "dtype": "float32"}, {"name": "conv2d/bias", "shape": [40], "dtype": "float32"}, {"name": "conv2d_1/kernel", "shape": [3, 3, 40, 60], "dtype": "float32"}, {"name": "conv2d_1/bias", "shape": [60], "dtype": "float32"}, {"name": "dense/kernel", "shape": [400, 22], "dtype": "float32"}, {"name": "dense/bias", "shape": [22], "dtype": "float32"}]}]};
var charset = ["", "0", "2", "4", "8", "A", "D", "G", "H", "J", "K", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "X", "Y"];
var model;

tf.setBackend('cpu'); // takes too long with webgl backend

function toggle(obj,v){
	if(v) obj.style.display = '';
	else obj.style.display = 'none';
}

function base64ToArray(base64) {
	var binary_string = window.atob(base64);
	var len = binary_string.length;
	var bytes = new Uint8Array(len);
	for (var i = 0; i < len; i++) {
			bytes[i] = binary_string.charCodeAt(i);
	}
	return bytes.buffer;
}

var iohander = {
	load: function(){
		return new Promise((resolve, reject) => {
			resolve({
				modelTopology: modelJSON.modelTopology,
				weightSpecs: modelJSON.weightsManifest[0]["weights"],
				weightData: base64ToArray(weights64),
				format: modelJSON.format,
				generatedBy: modelJSON.generatedBy,
				convertedBy: modelJSON.convertedBy
			});
		});
	}
}

async function load(){
	const uploadJSONInput = document.getElementById('upload-json');
	const uploadWeightsInput = document.getElementById('upload-weights-1');
	model = await tf.loadLayersModel(iohander);
	return model;
}

function black(x){ return x < 64; }

// Calculates "disorder" of the image. "Disorder" is the percentage of black pixels that have a
// non-black pixel below them. Minimizing this seems to be good enough metric for solving the slider.
function calculateDisorder(imgdata){
	var a = imgdata.data;
	var w = imgdata.width;
	var h = imgdata.height;
	var pic=[], visited=[];

	for(var c=0;c<w*h;c++){
		if(visited[c]) continue;
		if(! black(a[c*4])) continue;

		var blackCount = 0;
		var items = [];
		var toVisit = [c];
		while(toVisit.length>0){
			var cc = toVisit[toVisit.length-1];
			toVisit.splice(toVisit.length-1, 1);

			if(visited[cc]) continue;
			visited[cc]=1;

			if(black(a[cc*4])){
				items.push(cc);

				blackCount++;
				toVisit.push(cc+1);
				toVisit.push(cc-1);
				toVisit.push(cc+w);
				toVisit.push(cc-w);
			}
		}

		if(blackCount >= 24){
			items.forEach(function(x){ pic[x]=1; });
		}
	}

	var res = 0;
	var total = 0;
	for(var c=0;c<w*h-w;c++){
		if(pic[c]!=pic[c+w]) res+=1;
		if(pic[c]) total+=1;
	}

	return res / (total==0 ? 1 : total);
}

// returns ImageData from captcha's background image, foreground image, and offset (ranging from 0 to -50)
function imageFromCanvas(img, bg, off){
	var h=img.height, w=img.width;
	var th=80;
	var ph=0, pw=16;
	var scale = th/h

	var canvas = document.createElement('canvas');
	canvas.height = w * scale + pw*2;
	canvas.width = th;

	var ctx = canvas.getContext('2d');

	ctx.fillStyle = "rgb(238,238,238)";
	ctx.fillRect(0, 0, canvas.width, canvas.height);

	ctx.translate(canvas.width / 2, canvas.height / 2);
	ctx.scale(-scale,scale)
	ctx.rotate(90*Math.PI/180);

	var draw = function(off){
		if(bg){
			var border = 4;
			ctx.drawImage( bg, -off + border, 0, w-border*2, h, -w / 2 + border, -h / 2, w-border*2, h );
		}
		ctx.drawImage( img, -w / 2, -h / 2, w, h );
	}

	// if off is not specified and background image is present, try to figure out
	// the best offset automatically; select the offset that has smallest value of
	// calculateDisorder for the resulting image
	if(bg && off==null){
		var bestDisorder = 999;
		var bestImagedata = null;
		var bestOff=-1;

		for(var off=0;off>=-50;off--){
			draw(off);

			var imgdata = ctx.getImageData(0,0,canvas.width,canvas.height);
			var disorder = calculateDisorder(imgdata);

			if(disorder < bestDisorder){
				bestDisorder = disorder;
				bestImagedata = imgdata;
				bestOff = off;
			}
		}

		// not the best idea to do this here
		setTimeout(function(){
			var bg=document.getElementById('t-bg');
			var slider=document.getElementById('t-slider');
			if(! bg || !slider) return;

			slider.value = -bestOff*2;
			bg.style.backgroundPositionX = (bestOff)+"px";
		}, 1);

		return bestImagedata;
	} else{
		draw(off);

		return ctx.getImageData(0,0,canvas.width,canvas.height);
	}
}

// for debugging purposes
function imagedataToImage(imagedata) {
	var canvas = document.createElement('canvas');
	var ctx = canvas.getContext('2d');
	canvas.width = imagedata.width;
	canvas.height = imagedata.height;
	ctx.putImageData(imagedata, 0, 0);

	var image = new Image();
	image.src = canvas.toDataURL();
	return image;
}

async function predict(img, bg, off){
	if(! model){
		model = await load()
	}
	let image = imageFromCanvas(img, bg, off)
	let tensor = tf.browser.fromPixels(image, 1).mul(-1/238).add(1)
	let prediction = await model.predict(tensor.expandDims(0)).data()

	return createSequence(prediction)
}

function createSequence(prediction){
	var csl = charset.length
	let sequence = []

	for(var pos=0;pos<prediction.length;pos+=csl){
		var preds = prediction.slice(pos, pos+csl)
		var max = Math.max(...preds);

		var seqElem = {}

		for(var i=0;i<csl;i++){
			var p = preds[i] / max
			var c = charset[i+1]

			if(p>=0.05){
				seqElem[c||""] = p;
			}
		}

		sequence.push(seqElem)
	}

	return sequence
}

function postprocess(sequence, overrides){
	let csl = charset.length
	let possibilities = [{sequence: []}]

	sequence.forEach(function(e, i){
		var additions;
		if(overrides && overrides[i]!==undefined){
			additions = [{sym: overrides[i], off: i, conf: 1}]
		} else{
			additions = Object.keys(e).map(function(sym){ return {sym: sym, off: i, conf: e[sym]} });
		}

		if(additions.length==1 && additions[0].sym=='') return;

		let oldpos = possibilities
		possibilities = []
		oldpos.forEach(function(possibility){
			additions.forEach(function(a){
				var seq = [...possibility.sequence]
				if(a.sym != '') seq.push([a.sym, a.off, a.conf])

				var obj = {
					sequence: seq,
				}

				possibilities.push(obj)
			});
		});
	});

	var res = {}
	possibilities.forEach(function(p){
		var line = '';
		var lastSym = undefined;
		var lastOff = -1;
		var count = 0;
		var prob = 0;

		p.sequence.forEach(function(e){
			var sym=e[0];
			var off=e[1];
			var conf=e[2];

			if(sym==lastSym && lastOff + 2 >= off){
				return;
			}

			line += sym;

			lastSym = sym;
			lastOff = off;
			prob += conf;
			count++;
		});

		if(count>0) prob /= count;

		if(prob > res[line] || !res[line]){
			res[line] = prob;
		}
	});

	var keys = Object.keys(res).sort(function(a,b){ return res[a] < res[b]; })
	var keys_fitting = keys.filter(function(x){ return x.length==5 || x.length==6 })
	if(keys_fitting.length>0) keys = keys_fitting;

	return keys.map(function(x){ return {seq: x, prob: res[x]} })
}



async function imageFromUri(uri){
	if(uri.startsWith("url(\"")){
		uri = uri.substr(5,uri.length-7)
	}
	if(! uri.startsWith("data:")){
		return null
	}

	var img = new Image();
	await new Promise(r => img.onload=r, img.src=uri);

	return img
}


async function predictUri(uri, uribg, bgoff){
	var img = await imageFromUri(uri)
	var bg = uribg ? await imageFromUri(uribg) : null
	var off = bgoff ? parseInt(bgoff) : null;

	return await predict(img, bg, off)
}


var solveButton = document.createElement('input')
solveButton.id="t-auto-solve"
solveButton.value="Solve"
solveButton.type="button"
solveButton.style.fontSize = "11px"
solveButton.style.padding = "0 2px"
solveButton.style.margin = "0px 0px 0px 6px"
solveButton.style.height = "18px"
solveButton.onclick=async function(){ solve(true); }

var altsDiv = document.createElement('div')
altsDiv.id="t-auto-options"
altsDiv.style.margin = "0"
altsDiv.style.padding = "0"

var storedPalceholder;

var overrides = {}

function placeAfter(elem, sibling){
	if(elem.parentElement!=sibling.parentElement){
		setTimeout(function(){
			sibling.parentElement.insertBefore(elem,sibling.nextElementSibling);
		}, 1);
	}
}

var previousText = null;
async function solve(force){
	var resp = document.getElementById('t-resp');
	if(! resp) return;

	var bg=document.getElementById('t-bg');
	if(! bg) return;

	var fg=document.getElementById('t-fg');
	if(! fg) return;

	var help=document.getElementById('t-help');
	if(! help) return;

	placeAfter(solveButton, resp)
	placeAfter(altsDiv, help)

	// palememe
	setTimeout(function(){
		toggle(solveButton, bg.style.backgroundImage)
	}, 1);

	var text=fg.style.backgroundImage;
	if(! text){
		altsDiv.innerHTML = '';
		return;
	}

	if(text==previousText && !force) return;
	previousText=text;

	altsDiv.innerHTML = '';
	if(! storedPalceholder) storedPalceholder = resp.placeholder;
	resp.placeholder = "solving captcha...";

	overrides = {}

	var sequence = await predictUri(text, bg.style.backgroundImage, force ? bg.style.backgroundPositionX : null);
	var opts = postprocess(sequence);
	resp.placeholder = storedPalceholder;

	showOpts(opts);
}

function showOpts(opts){
	var resp = document.getElementById('t-resp');
	if(! resp) return;

	altsDiv.innerHTML = '';

	if(opts.length == 0){
		resp.value = '';
		return;
	}

	resp.value = opts[0].seq;

    // for now don't display options since it seems more difficult to pick than type the whole thing
	//if(opts.length == 1 || true)
		return;

	/*
	opts.forEach(function(opt, i){
		var span = document.createElement('span');
		span.style.padding="1px 3px";
		span.style.margin="2px";
		span.style.backgroundColor="#f4f4f4";
		span.style.cursor="pointer";
		span.style.borderRadius="3px";
		span.style.lineHeight="1.6em";
		span.textContent = opt.seq;
		span.title = (opt.prob*100).toFixed(0)+"%";
		span.addEventListener("click", function(){
			resp.value = opt.seq;
		});
		altsDiv.appendChild(span);
		altsDiv.appendChild(document.createTextNode(' '));
	});
	*/
}





var observer = new MutationObserver(async function(mutationsList, observer) {
	solve(false);
});



observer.observe(document.body, { attributes: true, childList: true, subtree: true });
