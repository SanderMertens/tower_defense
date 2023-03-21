uniform sampler2D atmos;
uniform vec3 u_sun_screen_pos;
uniform vec3 u_sun_color;
uniform float u_aspect;
uniform float u_sun_intensity;

in vec2 uv;
out vec4 frag_color;

void main() {
  vec2 sun_dist = uv.xy - u_sun_screen_pos.xy;
  sun_dist.x *= u_aspect;

  float sun_dist_len = length(sun_dist);
  float intensity = u_sun_intensity * float(u_sun_screen_pos.z >= 0.0);
  vec3 sunDisc = intensity * u_sun_color * ((sun_dist_len < 0.025) ? 1.0 : 0.0);

  frag_color = texture(atmos, uv) + vec4(sunDisc, 1.0);
}
