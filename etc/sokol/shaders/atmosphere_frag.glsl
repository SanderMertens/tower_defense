#include "etc/sokol/shaders/atmosphere.glsl"

uniform mat4 u_mat_v;
uniform vec3 u_eye_pos;
uniform vec3 u_light_pos;
uniform vec3 u_night_color;
uniform float u_aspect;
uniform float u_offset;

uniform float intensity;
uniform float planet_radius;
uniform float atmosphere_radius;
uniform vec3 rayleigh_coef;
uniform float mie_coef;
uniform float rayleigh_scale_height;
uniform float mie_scale_height;
uniform float mie_scatter_dir;

in vec2 uv;
out vec4 frag_color;

void main() {
  vec2 uv_scaled = uv * 1.3 - 0.66;
  uv_scaled.x *= u_aspect;
  uv_scaled.y += u_offset;

  vec4 coord = vec4(uv_scaled, -1, 0);
  vec3 ray = vec3(u_mat_v * coord);
  vec3 orig = vec3(u_eye_pos.x, 6372e3 + u_eye_pos.y, u_eye_pos.z);
  vec3 atmos = atmosphere(
      ray,                            // normalized ray direction
      orig,                           // ray origin
      -u_light_pos,                   // position of the sun
      intensity,                      // intensity of the sun
      planet_radius,                  // radius of the planet in meters
      atmosphere_radius,              // radius of the atmosphere in meters
      rayleigh_coef,                  // Rayleigh scattering coefficient
      mie_coef,                       // Mie scattering coefficient
      rayleigh_scale_height,          // Rayleigh scale height
      mie_scale_height,               // Mie scale height
      mie_scatter_dir                 // Mie preferred scattering direction
  );

  frag_color = max(vec4(atmos, 1.0), vec4(u_night_color, 1.0));
}
