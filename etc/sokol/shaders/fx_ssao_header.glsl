#include "etc/sokol/shaders/common.glsl"

// Based on https://threejs.org/examples/webgl_postprocessing_sao.html

// Increase/decrease to trade quality for performance
#define NUM_SAMPLES 3
// The sample kernel uses a spiral pattern so most samples are concentrated
// close to the center. 
#define NUM_RINGS 3
#define KERNEL_RADIUS 15.0
// Misc params, tweaked to match the renderer
#define BIAS 0.2
#define SCALE 0.1
// Derived constants
#define ANGLE_STEP ((PI2 * float(NUM_RINGS)) / float(NUM_SAMPLES))
#define INV_NUM_SAMPLES (1.0 / float(NUM_SAMPLES))

float getDepth(const in vec2 t_uv) {
    return rgba_to_depth(texture(t_depth, t_uv));
}

float getViewZ(const in float depth) {
    return (u_near * u_far) / (depth - u_far);
}

// Compute position in world space from depth & projection matrix
vec3 getViewPosition( const in vec2 screenPosition, const in float depth, const in float viewZ ) {
    float clipW = u_mat_p[2][3] * viewZ + u_mat_p[3][3];
    vec4 clipPosition = vec4( ( vec3( screenPosition, depth ) - 0.5 ) * 2.0, 1.0 );
    clipPosition *= clipW; // unprojection.
    return ( u_inv_mat_p * clipPosition ).xyz;
}

// Compute normal from derived position. Should at some point replace it
// with reading from a normal buffer so it works correctly with smooth
// shading / normal maps.
vec3 getViewNormal( const in vec3 viewPosition, const in vec2 t_uv ) {
    return normalize( cross( dFdx( viewPosition ), dFdy( viewPosition ) ) );
}

float scaleDividedByCameraFar;

// Compute occlusion of single sample
float getOcclusion( const in vec3 centerViewPosition, const in vec3 centerViewNormal, const in vec3 sampleViewPosition ) {
    vec3 viewDelta = sampleViewPosition - centerViewPosition;
    float viewDistance = length( viewDelta );
    float scaledScreenDistance = scaleDividedByCameraFar * viewDistance;
    float n_dot_d = dot(centerViewNormal, viewDelta);
    float scaled_n_dot_d = max(0.0, n_dot_d / scaledScreenDistance - BIAS);
    float result = scaled_n_dot_d / (1.0 + pow2(scaledScreenDistance));

    // Strip off values that are too large which eliminates shadowing objects
    // that are far away.
    if (result > 220.0) {
      result = 0.0;
    }

    // Squash the range and offset noise.
    return max(0.0, clamp(result, 1.1, 20.0) / 13.0 - 0.2);
}

float getAmbientOcclusion( const in vec3 centerViewPosition, float centerDepth ) {
  scaleDividedByCameraFar = SCALE / u_far;
  vec3 centerViewNormal = getViewNormal( centerViewPosition, uv );

  float angle = rand( uv ) * PI2;
  vec2 radius = vec2( KERNEL_RADIUS * INV_NUM_SAMPLES );

  // Use smaller kernels for objects farther away from the camera
  radius /= u_target_size * centerDepth * 0.05;
  // Make sure tha the sample radius isn't less than a single texel, as this
  // introduces noise
  radius = max(radius, 5.0 / u_target_size);

  vec2 radiusStep = radius;
  float occlusionSum = 0.0;

  // Collect occlusion samples
  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    vec2 sampleUv = uv + vec2( cos( angle ), sin( angle ) ) * radius;

    // Don't sample outside of texture coords to prevent edge artifacts
    sampleUv = clamp(sampleUv, EPSILON, 1.0 - EPSILON);

    radius += radiusStep;
    angle += ANGLE_STEP;

    float sampleDepth = getDepth( sampleUv );
    float sampleDepthNorm = sampleDepth / u_far;

    float sampleViewZ = getViewZ( sampleDepth );
    vec3 sampleViewPosition = getViewPosition( sampleUv, sampleDepthNorm, sampleViewZ );
    occlusionSum += getOcclusion( centerViewPosition, centerViewNormal, sampleViewPosition );
  }

  return occlusionSum * (1.0 / (float(NUM_SAMPLES)));
}
