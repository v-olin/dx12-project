// render as 600x400
#include "weeping_willow.inc"
background {rgb <0.95,0.95,0.9>}
light_source { <5000,5000,-3000>, rgb 1.2 }
light_source { <-5000,2000,3000>, rgb 0.5 shadowless }
#declare HEIGHT = weeping_willow_13_height * 1.3;
#declare WIDTH = HEIGHT*400/600;
camera { orthographic location <0, HEIGHT*0.45, -100>
         right <WIDTH, 0, 0> up <0, HEIGHT, 0>
         look_at <0, HEIGHT*0.45, -80> }
union { 
         object { weeping_willow_13_stems
                pigment {color rgb 0.9} }
         object { weeping_willow_13_leaves
                texture { pigment {color rgb 1} 
                          finish { ambient 0.15 diffuse 0.8 }}}
         rotate 90*y }
