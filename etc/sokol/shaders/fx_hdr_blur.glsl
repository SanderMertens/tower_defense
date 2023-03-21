const float gauss[5] = float[] (0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

float offset = 0.0;
vec4 result = gauss[0] * texture(tex, vec2(uv.x, uv.y));
ivec2 ts = textureSize(tex, 0);
float px = 1.0 / float(ts.x);
float py = 1.0 / float(ts.y);
if (horizontal == 0.0) {
   for (int i = 1; i < 5; i ++) {
     offset += px / u_aspect;
     result += gauss[i] * texture(tex, vec2(uv.x + offset, uv.y));
     result += gauss[i] * texture(tex, vec2(uv.x - offset, uv.y));
   }
} else {
   for (int i = 1; i < 5; i ++) {
     offset += py;
     result += gauss[i] * texture(tex, vec2(uv.x, uv.y + offset));
     result += gauss[i] * texture(tex, vec2(uv.x, uv.y - offset));
   }
}

frag_color = result;
