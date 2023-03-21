
const vec3 channel_lum = vec3(0.299, 0.587, 0.114);
const float lmax = 0.2 / dot(vec3(1.0, 1.0, 1.0), channel_lum);

vec4 c = texture(hdr, uv, mipmap);
float l = dot(c.rgb, channel_lum);
l = l * lmax;
frag_color = c * l;
