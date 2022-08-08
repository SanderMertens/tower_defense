#include "flecs_game.h"

ECS_DECLARE(EcsCameraController);

#define CAMERA_DECELERATION 100.0
#define CAMERA_ANGULAR_DECELERATION 5.0

static const float CameraDeceleration = CAMERA_DECELERATION;
static const float CameraAcceleration = 50.0 + CAMERA_DECELERATION;
static const float CameraAngularDeceleration = CAMERA_ANGULAR_DECELERATION;
static const float CameraAngularAcceleration = 2.5 + CAMERA_ANGULAR_DECELERATION;
static const float CameraMaxSpeed = 30.0;

static
void CameraControllerAddPosition(ecs_iter_t *it) {
    for (int i = 0; i < it->count; i ++) {
        ecs_set(it->world, it->entities[i], EcsPosition3, {0, -1.5});
    }
}

static
void CameraControllerAddRotation(ecs_iter_t *it) {
    for (int i = 0; i < it->count; i ++) {
        ecs_set(it->world, it->entities[i], EcsRotation3, {0, 0});
    }
}

static
void CameraControllerAddVelocity(ecs_iter_t *it) {
    for (int i = 0; i < it->count; i ++) {
        ecs_set(it->world, it->entities[i], EcsVelocity3, {0, 0});
    }
}

static
void CameraControllerAddAngularVelocity(ecs_iter_t *it) {
    for (int i = 0; i < it->count; i ++) {
        ecs_set(it->world, it->entities[i], EcsAngularVelocity, {0, 0});
    }
}

static
void CameraControllerSyncPosition(ecs_iter_t *it) {
    EcsCamera *camera = ecs_field(it, EcsCamera, 1);
    EcsPosition3 *p = ecs_field(it, EcsPosition3, 2);

    for (int i = 0; i < it->count; i ++) {
        camera[i].position[0] = p[i].x;
        camera[i].position[1] = p[i].y;
        camera[i].position[2] = p[i].z;
    }
}

static
void CameraControllerSyncRotation(ecs_iter_t *it) {
    EcsCamera *camera = ecs_field(it, EcsCamera, 1);
    EcsPosition3 *p = ecs_field(it, EcsPosition3, 2);
    EcsRotation3 *r = ecs_field(it, EcsRotation3, 3);

    for (int i = 0; i < it->count; i ++) {
        camera[i].lookat[0] = p[i].x + sin(r[i].y) * cos(r[i].x);
        camera[i].lookat[1] = p[i].y + sin(r[i].x);
        camera[i].lookat[2] = p[i].z + cos(r[i].y) * cos(r[i].x);;
    }
}

static
void CameraControllerAccelerate(ecs_iter_t *it) {
    EcsInput *input = ecs_field(it, EcsInput, 1);
    EcsRotation3 *r = ecs_field(it, EcsRotation3, 2);
    EcsVelocity3 *v = ecs_field(it, EcsVelocity3, 3);
    EcsAngularVelocity *av = ecs_field(it, EcsAngularVelocity, 4);

    for (int i = 0; i < it->count; i ++) {
        float angle = r[i].y;
        float accel = CameraAcceleration * it->delta_time;
        float angular_accel = CameraAngularAcceleration * it->delta_time;

        // Camera XZ movement
        if (input->keys[ECS_KEY_W].state) {
            v[i].x += sin(angle) * accel;
            v[i].z += cos(angle) * accel;
        }
        if (input->keys[ECS_KEY_S].state) {
            v[i].x += sin(angle + GLM_PI) * accel;
            v[i].z += cos(angle + GLM_PI) * accel;
        }

        if (input->keys[ECS_KEY_D].state) {
            v[i].x += cos(angle) * accel;
            v[i].z -= sin(angle) * accel;
        }
        if (input->keys[ECS_KEY_A].state) {
            v[i].x += cos(angle + GLM_PI) * accel;
            v[i].z -= sin(angle + GLM_PI) * accel;
        }

        // Camera Y movement
        if (input->keys[ECS_KEY_E].state) {
            v[i].y -= accel;
        }
        if (input->keys[ECS_KEY_Q].state) {
            v[i].y += accel;
        }

        // Camera Y rotation
        if (input->keys[ECS_KEY_RIGHT].state) {
            av[i].y += angular_accel;
        }
        if (input->keys[ECS_KEY_LEFT].state) {
            av[i].y -= angular_accel;
        }

        // Camera X rotation
        if (input->keys[ECS_KEY_UP].state) {
            av[i].x -= angular_accel;
        }
        if (input->keys[ECS_KEY_DOWN].state) {
            av[i].x += angular_accel;
        }
    }
}

static
void camera_controller_decel(float *v_ptr, float a, float dt) {
    float v = v_ptr[0];

    if (v > 0) {
        v = glm_clamp(v - a * dt, 0, v);
    }
    if (v < 0) {
        v = glm_clamp(v + a * dt, v, 0);
    }

    v_ptr[0] = v;
}

static
void CameraControllerDecelerate(ecs_iter_t *it) {
    EcsVelocity3 *v = ecs_field(it, EcsVelocity3, 1);
    EcsAngularVelocity *av = ecs_field(it, EcsAngularVelocity, 2);
    EcsRotation3 *r = ecs_field(it, EcsRotation3, 3);

    float dt = it->delta_time;

    vec3 zero = {0};

    for (int i = 0; i < it->count; i ++) {
        vec3 v3 = {v[i].x, v[i].y, v[i].z}, vn3;
        glm_vec3_normalize_to(v3, vn3);
        
        float speed = glm_vec3_distance(zero, v3);
        if (speed > CameraMaxSpeed) {
            glm_vec3_scale(v3, CameraMaxSpeed / speed, v3);
            v[i].x = v3[0];
            v[i].y = v3[1];
            v[i].z = v3[2];
        }

        camera_controller_decel(&v[i].x, CameraDeceleration * fabs(vn3[0]), dt);
        camera_controller_decel(&v[i].y, CameraDeceleration * fabs(vn3[1]), dt);
        camera_controller_decel(&v[i].z, CameraDeceleration * fabs(vn3[2]), dt);

        camera_controller_decel(&av[i].x, CameraAngularDeceleration, dt);
        camera_controller_decel(&av[i].y, CameraAngularDeceleration, dt);

        if (r[i].x > M_PI / 2.0) {
            r[i].x = M_PI / 2.0 - 0.0001;
            av[i].x = 0;
        }
        if (r[i].x < -M_PI / 2.0) {
            r[i].x = -(M_PI / 2.0) + 0.0001;
            av[i].x = 0;
        }
    }
}

void FlecsGameImport(ecs_world_t *world) {
    ECS_MODULE(world, FlecsGame);

    ECS_IMPORT(world, FlecsComponentsTransform);
    ECS_IMPORT(world, FlecsComponentsPhysics);
    ECS_IMPORT(world, FlecsComponentsGraphics);
    ECS_IMPORT(world, FlecsComponentsInput);
    ECS_IMPORT(world, FlecsSystemsPhysics);

    ecs_set_name_prefix(world, "Ecs");

    ECS_TAG_DEFINE(world, EcsCameraController);

    ECS_SYSTEM(world, CameraControllerAddPosition, EcsOnLoad,
        [filter] flecs.components.graphics.Camera,
        [filter] CameraController,
        [out]    !flecs.components.transform.Position3);

    ECS_SYSTEM(world, CameraControllerAddRotation, EcsOnLoad,
        [filter] flecs.components.graphics.Camera,
        [filter] CameraController,
        [out]    !flecs.components.transform.Rotation3);

    ECS_SYSTEM(world, CameraControllerAddVelocity, EcsOnLoad,
        [filter] flecs.components.graphics.Camera,
        [filter] CameraController,
        [out]    !flecs.components.physics.Velocity3);

    ECS_SYSTEM(world, CameraControllerAddAngularVelocity, EcsOnLoad,
        [filter] flecs.components.graphics.Camera,
        [filter] CameraController,
        [out]    !flecs.components.physics.AngularVelocity);

    ECS_SYSTEM(world, CameraControllerSyncPosition, EcsOnUpdate,
        [out]    flecs.components.graphics.Camera, 
        [in]     flecs.components.transform.Position3,
        [filter] CameraController);

    ECS_SYSTEM(world, CameraControllerSyncRotation, EcsOnUpdate,
        [out]    flecs.components.graphics.Camera, 
        [in]     flecs.components.transform.Position3,
        [in]     flecs.components.transform.Rotation3,
        [filter] CameraController);

    ECS_SYSTEM(world, CameraControllerAccelerate, EcsOnUpdate,
        [in]     flecs.components.input.Input($),
        [in]     flecs.components.transform.Rotation3,
        [inout]  flecs.components.physics.Velocity3,
        [inout]  flecs.components.physics.AngularVelocity,
        [filter] CameraController);

    ECS_SYSTEM(world, CameraControllerDecelerate, EcsOnUpdate,
        [inout]  flecs.components.physics.Velocity3,
        [inout]  flecs.components.physics.AngularVelocity,
        [inout]  flecs.components.transform.Rotation3,
        [filter] CameraController);
}

