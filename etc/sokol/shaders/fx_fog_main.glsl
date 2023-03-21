#define LOG2 1.442695

const float transition = 0.025;
const float inv_transition = 1.0 / transition;
const float sample_height = 0.01;

vec4 c = texture(hdr, uv);
float d = (rgba_to_depth(texture(depth, uv)) / u_far);
vec4 fog_color = texture(atmos, vec2(uv.x, u_horizon + sample_height));
float intensity;
if (d > 1.0) {
    intensity = (max((u_horizon + transition) - uv.y, 0.0) * inv_transition);
    intensity = min(intensity, 1.0);
} else {
    intensity = 1.0 - exp2(-(d * d) * u_density * u_density * LOG2);
}
frag_color = mix(c, fog_color, intensity);
