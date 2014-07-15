## MSharpen ##

MSharpen is a very simple masked sharpening plugin for AviSynth. This version is a reimplementation of the old neuron2's plugin. It features better performance, x64 compatibility and less bugs.

### Usage
```
MSharpen(clip c, int "threshold", int "strength", bool "highq", bool "mask")
```
* *threshold (0-255, default 10)* - determines what is detected as edge detail and thus sharpened. To see what edge detail areas will be sharpened, use the 'mask' parameter.
* *strength (0-255, default 100)* - strength of the sharpening to be applied to the edge detail areas. It is applied only to the edge detail areas as determined by the *threshold* parameter. Strength 255 is the strongest sharpening.
* *mask (default false)*: When set to true, the areas to be sharpened are shown in white against a black background. Use this to set the level of detail to be sharpened. This function also makes a basic edge detection filter.
* *highq (default true)* - lets you tradeoff speed for quality of detail detection. Set it to true for the best detail detection. Set it to false for maximum speed.
