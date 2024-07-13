#define FLECS_GAME_IMPL

#include "flecs_game.h"

#define VARIATION_SLOTS_MAX (20)

ECS_DECLARE(EcsCameraController);

void FlecsGameCameraControllerImport(ecs_world_t *world);
void FlecsGameLightControllerImport(ecs_world_t *world);

ECS_CTOR(EcsParticleEmitter, ptr, {
    ptr->particle = 0;
    ptr->spawn_interval = 1.0;
    ptr->lifespan = 1000.0;
    ptr->size_decay = 1.0;
    ptr->color_decay = 1.0;
    ptr->velocity_decay = 1.0;
    ptr->t = 0;
})

static
float randf(float max) {
    return max * (float)rand() / (float)RAND_MAX;
}

typedef struct {
    float x_count;
    float y_count;
    float z_count;
    float x_spacing;
    float y_spacing;
    float z_spacing;
    float x_half;
    float y_half;
    float z_half;
    float x_var;
    float y_var;
    float z_var;
    float variations_total;
    int32_t variations_count;
    ecs_entity_t variations[VARIATION_SLOTS_MAX];
    ecs_entity_t prefab;
} flecs_grid_params_t;

static
ecs_entity_t get_prefab(
    ecs_world_t *world, 
    ecs_entity_t parent,
    ecs_entity_t prefab) 
{
    if (!prefab) {
        return 0;
    }

    /* If prefab is a script/assembly, create a private instance of the
     * assembly for the grid with default values. This allows applications to
     * use assemblies directly vs. having to create a dummy prefab */
    ecs_entity_t result = prefab;
    if (ecs_has(world, prefab, EcsScript) && ecs_has(world, prefab, EcsComponent)) {
        result = ecs_new(world);
        ecs_add_id(world, result, EcsPrefab);
        ecs_add_id(world, result, prefab);
    }

    return result;
}

static
ecs_entity_t generate_tile(
    ecs_world_t *world,
    const EcsGrid *grid,
    float xc,
    float yc,
    float zc,
    const flecs_grid_params_t *params)
{
    if (params->x_var) {
        xc += randf(params->x_var) - params->x_var / 2;
    }
    if (params->y_var) {
        yc += randf(params->y_var) - params->y_var / 2;
    }
    if (params->z_var) {
        zc += randf(params->z_var) - params->z_var / 2;
    }

    ecs_entity_t slot = 0;
    if (params->prefab) {
        slot = params->prefab;
    } else {
        float p = randf(params->variations_total), cur = 0;
        for (int v = 0; v < params->variations_count; v ++) {
            cur += grid->variations[v].chance;
            if (p <= cur) {
                slot = params->variations[v];
                break;
            }
        }
    }

    ecs_entity_t inst = ecs_new_w_pair(world, EcsIsA, slot);
    ecs_set(world, inst, EcsPosition3, {xc, yc, zc});
    return inst;
}

static
void generate_grid(
    ecs_world_t *world, 
    ecs_entity_t parent, 
    const EcsGrid *grid) 
{
    flecs_grid_params_t params = {0};

    params.x_count = glm_max(1, grid->x.count);
    params.y_count = glm_max(1, grid->y.count);
    params.z_count = glm_max(1, grid->z.count);

    bool border = false;
    if (grid->border.x || grid->border.y || grid->border.z) {
        params.x_spacing = grid->border.x / params.x_count;
        params.y_spacing = grid->border.y / params.y_count;
        params.z_spacing = grid->border.z / params.z_count;
        border = true;
    } else {
        params.x_spacing = glm_max(0.001, grid->x.spacing);
        params.y_spacing = glm_max(0.001, grid->y.spacing);
        params.z_spacing = glm_max(0.001, grid->z.spacing);
    }

    params.x_half = ((params.x_count - 1) / 2.0) * params.x_spacing;
    params.y_half = ((params.y_count - 1) / 2.0) * params.y_spacing;
    params.z_half = ((params.z_count - 1) / 2.0) * params.z_spacing;
    
    params.x_var = grid->x.variation;
    params.y_var = grid->y.variation;
    params.z_var = grid->z.variation;

    ecs_entity_t old_scope = ecs_set_scope(world, parent);

    ecs_entity_t prefab = grid->prefab;
    params.variations_total = 0;
    params.variations_count = 0;
    if (!prefab) {
        for (int i = 0; i < VARIATION_SLOTS_MAX; i ++) {
            if (!grid->variations[i].prefab) {
                break;
            }
            params.variations[i] = get_prefab(world, parent, 
                grid->variations[i].prefab);
            params.variations_total += grid->variations[i].chance;
            params.variations_count ++;
        }
    } else {
        prefab = params.prefab = get_prefab(world, parent, prefab);
    }

    if (!prefab && !params.variations_count) {
        return;
    }

    if (!border) {
        for (int32_t x = 0; x < params.x_count; x ++) {
            for (int32_t y = 0; y < params.y_count; y ++) {
                for (int32_t z = 0; z < params.z_count; z ++) {
                    float xc = (float)x * params.x_spacing - params.x_half;
                    float yc = (float)y * params.y_spacing - params.y_half;
                    float zc = (float)z * params.z_spacing - params.z_half;
                    generate_tile(world, grid, xc, yc, zc, &params);
                }
            }
        }
    } else {
        for (int32_t x = 0; x < params.x_count; x ++) {
            float xc = (float)x * params.x_spacing - params.x_half;
            float zc = grid->border.z / 2 + grid->border_offset.z;
            generate_tile(world, grid, xc, 0, -zc, &params);
            generate_tile(world, grid, xc, 0, zc, &params);
        }

        for (int32_t x = 0; x < params.z_count; x ++) {
            float xc = grid->border.x / 2 + grid->border_offset.x;
            float zc = (float)x * params.z_spacing - params.z_half;
            ecs_entity_t inst;
            inst = generate_tile(world, grid, xc, 0, zc, &params);
            ecs_set(world, inst, EcsRotation3, {0, GLM_PI / 2, 0});
            inst = generate_tile(world, grid, -xc, 0, zc, &params);
            ecs_set(world, inst, EcsRotation3, {0, GLM_PI / 2, 0});
        }
    }

    ecs_set_scope(world, old_scope);
}

static
void SetGrid(ecs_iter_t *it) {
    EcsGrid *grid = ecs_field(it, EcsGrid, 0);

    for (int i = 0; i < it->count; i ++) {
        ecs_entity_t g = it->entities[0];
        ecs_delete_with(it->world, ecs_pair(EcsChildOf, g));
        generate_grid(it->world, g, &grid[i]);
    }
}

static
void ParticleEmit(ecs_iter_t *it) {
    EcsParticleEmitter *e = ecs_field(it, EcsParticleEmitter, 0);
    EcsBox *box = ecs_field(it, EcsBox, 1);

    for (int i = 0; i < it->count; i ++) {
        e[i].t += it->delta_time;
        if (e[i].t > e[i].spawn_interval) {
            e[i].t -= e[i].spawn_interval;

            ecs_entity_t p = ecs_insert(it->world, 
                {ecs_childof(it->entities[i])},
                {ecs_isa(e[i].particle)},
                ecs_value(EcsParticle, {
                    .t = e[i].lifespan
                }));

            if (box) {
                EcsPosition3 pos = {0, 0, 0};
                pos.x = randf(box[i].width) - box[i].width / 2;
                pos.y = randf(box[i].height) - box[i].height / 2;
                pos.z = randf(box[i].depth) - box[i].depth / 2;
                ecs_set_ptr(it->world, p, EcsPosition3, &pos);
            }

            ecs_set(it->world, p, EcsRotation3, { 0, randf(4 * 3.1415926), 0 });
        }
    }
}

static
void ParticleProgress(ecs_iter_t *it) {
    EcsParticle *p = ecs_field(it, EcsParticle, 0);
    EcsParticleEmitter *e = ecs_field(it, EcsParticleEmitter, 1);
    EcsBox *box = ecs_field(it, EcsBox, 2);
    EcsRgb *color = ecs_field(it, EcsRgb, 3);
    EcsVelocity3 *vel = ecs_field(it, EcsVelocity3, 4);

    for (int i = 0; i < it->count; i ++) {
        p[i].t -= it->delta_time;
        if (p[i].t <= 0) {
            ecs_delete(it->world, it->entities[i]);
        }
    }

    if (box) {
        for (int i = 0; i < it->count; i ++) {
            box[i].width *= pow(e[i].size_decay, it->delta_time);
            box[i].height *= pow(e[i].size_decay, it->delta_time);
            box[i].depth *= pow(e[i].size_decay, it->delta_time);

            if ((box[i].width + box[i].height + box[i].depth) < 0.1) {
                ecs_delete(it->world, it->entities[i]);
            }
        }
    }
    if (color) {
        for (int i = 0; i < it->count; i ++) {
            color[i].r *= pow(e[i].color_decay, it->delta_time);
            color[i].g *= pow(e[i].color_decay, it->delta_time);
            color[i].b *= pow(e[i].color_decay, it->delta_time);
        } 
    }
    if (vel) {
        for (int i = 0; i < it->count; i ++) {
            vel[i].x *= pow(e[i].velocity_decay, it->delta_time);
            vel[i].y *= pow(e[i].velocity_decay, it->delta_time);
            vel[i].z *= pow(e[i].velocity_decay, it->delta_time);
        }
    }
}

void FlecsGameImport(ecs_world_t *world) {
    ECS_MODULE(world, FlecsGame);

    ECS_IMPORT(world, FlecsComponentsTransform);
    ECS_IMPORT(world, FlecsComponentsPhysics);
    ECS_IMPORT(world, FlecsComponentsGraphics);
    ECS_IMPORT(world, FlecsComponentsGui);
    ECS_IMPORT(world, FlecsComponentsInput);
    ECS_IMPORT(world, FlecsSystemsPhysics);

    ecs_set_name_prefix(world, "Ecs");

    ECS_TAG_DEFINE(world, EcsCameraController);
    ECS_META_COMPONENT(world, EcsCameraAutoMove);
    ECS_META_COMPONENT(world, EcsTimeOfDay);
    ECS_META_COMPONENT(world, ecs_grid_slot_t);
    ECS_META_COMPONENT(world, ecs_grid_coord_t);
    ECS_META_COMPONENT(world, EcsGrid);
    ECS_META_COMPONENT(world, EcsParticleEmitter);
    ECS_META_COMPONENT(world, EcsParticle);

    FlecsGameCameraControllerImport(world);
    FlecsGameLightControllerImport(world);

    ecs_set_hooks(world, EcsTimeOfDay, {
        .ctor = flecs_default_ctor
    });

    ECS_OBSERVER(world, SetGrid, EcsOnSet, Grid);

    ecs_set_hooks(world, EcsParticleEmitter, {
        .ctor = ecs_ctor(EcsParticleEmitter)
    });

    ECS_SYSTEM(world, ParticleEmit, EcsOnUpdate, 
        ParticleEmitter,
        ?flecs.components.geometry.Box);

    ECS_SYSTEM(world, ParticleProgress, EcsOnUpdate, 
        Particle, 
        ParticleEmitter(up),
        ?flecs.components.geometry.Box(self),
        ?flecs.components.graphics.Rgb(self),
        ?flecs.components.physics.Velocity3(self));
}


#define CAMERA_DECELERATION 100.0
#define CAMERA_ANGULAR_DECELERATION 5.0

static const float CameraDeceleration = CAMERA_DECELERATION;
static const float CameraAcceleration = 50.0 + CAMERA_DECELERATION;
static const float CameraAngularDeceleration = CAMERA_ANGULAR_DECELERATION;
static const float CameraAngularAcceleration = 2.5 + CAMERA_ANGULAR_DECELERATION;
static const float CameraMaxSpeed = 40.0;

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
    EcsCamera *camera = ecs_field(it, EcsCamera, 0);
    EcsPosition3 *p = ecs_field(it, EcsPosition3, 1);

    for (int i = 0; i < it->count; i ++) {
        camera[i].position[0] = p[i].x;
        camera[i].position[1] = p[i].y;
        camera[i].position[2] = p[i].z;
    }
}

static
void CameraControllerSyncRotation(ecs_iter_t *it) {
    EcsCamera *camera = ecs_field(it, EcsCamera, 0);
    EcsPosition3 *p = ecs_field(it, EcsPosition3, 1);
    EcsRotation3 *r = ecs_field(it, EcsRotation3, 2);

    for (int i = 0; i < it->count; i ++) {
        camera[i].lookat[0] = p[i].x + sin(r[i].y) * cos(r[i].x);
        camera[i].lookat[1] = p[i].y + sin(r[i].x);
        camera[i].lookat[2] = p[i].z + cos(r[i].y) * cos(r[i].x);;
    }
}

static
void CameraControllerSyncLookAt(ecs_iter_t *it) {
    EcsCamera *camera = ecs_field(it, EcsCamera, 0);
    EcsLookAt *lookat = ecs_field(it, EcsLookAt, 1);

    for (int i = 0; i < it->count; i ++) {
        camera[i].lookat[0] = lookat[i].x;
        camera[i].lookat[1] = lookat[i].y;
        camera[i].lookat[2] = lookat[i].z;
    }
}

static
void CameraControllerAccelerate(ecs_iter_t *it) {
    EcsInput *input = ecs_field(it, EcsInput, 0);
    EcsRotation3 *r = ecs_field(it, EcsRotation3, 1);
    EcsVelocity3 *v = ecs_field(it, EcsVelocity3, 2);
    EcsAngularVelocity *av = ecs_field(it, EcsAngularVelocity, 3);

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
            v[i].y += accel;
        }
        if (input->keys[ECS_KEY_Q].state) {
            v[i].y -= accel;
        }

        // Camera Y rotation
        if (input->keys[ECS_KEY_LEFT].state) {
            av[i].y -= angular_accel;
        }
        if (input->keys[ECS_KEY_RIGHT].state) {
            av[i].y += angular_accel;
        }

        // Camera X rotation
        if (input->keys[ECS_KEY_UP].state) {
            av[i].x += angular_accel;
        }
        if (input->keys[ECS_KEY_DOWN].state) {
            av[i].x -= angular_accel;
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
    EcsVelocity3 *v = ecs_field(it, EcsVelocity3, 0);
    EcsAngularVelocity *av = ecs_field(it, EcsAngularVelocity, 1);
    EcsRotation3 *r = ecs_field(it, EcsRotation3, 2);

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

        if (r[i].x > GLM_PI / 2.0) {
            r[i].x = GLM_PI / 2.0 - 0.0001;
            av[i].x = 0;
        }
        if (r[i].x < -GLM_PI / 2.0) {
            r[i].x = -(GLM_PI / 2.0) + 0.0001;
            av[i].x = 0;
        }
    }
}

static
void CameraAutoMove(ecs_iter_t *it) {
    EcsVelocity3 *v = ecs_field(it, EcsVelocity3, 0);
    EcsCameraAutoMove *m = ecs_field(it, EcsCameraAutoMove, 1);

    float dt = it->delta_time;

    for (int i = 0; i < it->count; i ++) {
        EcsVelocity3 *vcur = &v[i];
        m->t += dt;
        if ((m->t < m->after) && (vcur->x || vcur->y || vcur->z)) {
            m->t = 0;
        }
        if (m->t > m->after) {
            vcur->z = 10;
        }
    }
}

void FlecsGameCameraControllerImport(ecs_world_t *world) {
    ECS_SYSTEM(world, CameraControllerAddPosition, EcsOnLoad,
        [none]   flecs.components.graphics.Camera,
        [none]   CameraController,
        [out]    !flecs.components.transform.Position3);

    ECS_SYSTEM(world, CameraControllerAddRotation, EcsOnLoad,
        [none]   flecs.components.graphics.Camera,
        [none]   CameraController,
        [out]    !flecs.components.transform.Rotation3);

    ECS_SYSTEM(world, CameraControllerAddVelocity, EcsOnLoad,
        [none]   flecs.components.graphics.Camera,
        [none]   CameraController,
        [out]    !flecs.components.physics.Velocity3);

    ECS_SYSTEM(world, CameraControllerAddAngularVelocity, EcsOnLoad,
        [none]   flecs.components.graphics.Camera,
        [none]   CameraController,
        [out]    !flecs.components.physics.AngularVelocity);

    ECS_SYSTEM(world, CameraControllerSyncPosition, EcsOnUpdate,
        [out]    flecs.components.graphics.Camera, 
        [in]     flecs.components.transform.Position3,
        [none]   CameraController);

    ECS_SYSTEM(world, CameraControllerSyncRotation, EcsOnUpdate,
        [out]    flecs.components.graphics.Camera, 
        [in]     flecs.components.transform.Position3,
        [in]     flecs.components.transform.Rotation3,
        [none]   CameraController);

    ECS_SYSTEM(world, CameraControllerSyncLookAt, EcsOnUpdate,
        [out]    flecs.components.graphics.Camera, 
        [in]     flecs.components.graphics.LookAt,
        [none]   CameraController);

    ECS_SYSTEM(world, CameraControllerAccelerate, EcsOnUpdate,
        [in]     flecs.components.input.Input($),
        [in]     flecs.components.transform.Rotation3,
        [inout]  flecs.components.physics.Velocity3,
        [inout]  flecs.components.physics.AngularVelocity,
        [none]   CameraController);

    ECS_SYSTEM(world, CameraControllerDecelerate, EcsOnUpdate,
        [inout]  flecs.components.physics.Velocity3,
        [inout]  flecs.components.physics.AngularVelocity,
        [inout]  flecs.components.transform.Rotation3,
        [none]   CameraController);

    ECS_SYSTEM(world, CameraAutoMove, EcsOnUpdate,
        [inout]  flecs.components.physics.Velocity3,
        [inout]  CameraAutoMove);
}


static
void LightControllerSyncPosition(ecs_iter_t *it) {
    EcsDirectionalLight *light = ecs_field(it, EcsDirectionalLight, 0);
    EcsPosition3 *p = ecs_field(it, EcsPosition3, 1);

    for (int i = 0; i < it->count; i ++) {
        light[i].position[0] = p[i].x;
        light[i].position[1] = p[i].y;
        light[i].position[2] = p[i].z;
    }
}

static
void LightControllerSyncRotation(ecs_iter_t *it) {
    EcsDirectionalLight *light = ecs_field(it, EcsDirectionalLight, 0);
    EcsPosition3 *p = ecs_field(it, EcsPosition3, 1);
    EcsRotation3 *r = ecs_field(it, EcsRotation3, 2);

    for (int i = 0; i < it->count; i ++) {
        light[i].direction[0] = p[i].x + sin(r[i].y) * cos(r[i].x);
        light[i].direction[1] = p[i].y + sin(r[i].x);
        light[i].direction[2] = p[i].z + cos(r[i].y) * cos(r[i].x);
    }
}

static
void LightControllerSyncColor(ecs_iter_t *it) {
    EcsDirectionalLight *light = ecs_field(it, EcsDirectionalLight, 0);
    EcsRgb *color = ecs_field(it, EcsRgb, 1);

    for (int i = 0; i < it->count; i ++) {
        light[i].color[0] = color[i].r;
        light[i].color[1] = color[i].g;
        light[i].color[2] = color[i].b;
    }
}

static
void LightControllerSyncIntensity(ecs_iter_t *it) {
    EcsDirectionalLight *light = ecs_field(it, EcsDirectionalLight, 0);
    EcsLightIntensity *intensity = ecs_field(it, EcsLightIntensity, 1);

    for (int i = 0; i < it->count; i ++) {
        light[i].intensity = intensity[i].value;
    }
}

static
void TimeOfDayUpdate(ecs_iter_t *it) {
    EcsTimeOfDay *tod = ecs_field(it, EcsTimeOfDay, 0);
    tod->t += it->delta_time * tod->speed;
}

static
float get_time_of_day(float t) {
    return (t + 1.0) * GLM_PI;
}

static
float get_sun_height(float t) {
    return -sin(get_time_of_day(t));
}

static
void LightControllerTimeOfDay(ecs_iter_t *it) {
    EcsTimeOfDay *tod = ecs_field(it, EcsTimeOfDay, 0);
    EcsRotation3 *r = ecs_field(it, EcsRotation3, 1);
    EcsRgb *color = ecs_field(it, EcsRgb, 2);
    EcsLightIntensity *light_intensity = ecs_field(it, EcsLightIntensity, 3);

    static vec3 day = {0.8, 0.8, 0.75};
    static vec3 twilight = {1.0, 0.1, 0.01};
    float twilight_angle = 0.3;

    for (int i = 0; i < it->count; i ++) {
        r[i].x = get_time_of_day(tod[i].t);

        float t_sin = get_sun_height(tod[i].t);
        float t_sin_low = twilight_angle - t_sin;
        vec3 sun_color;
        if (t_sin_low > 0) {
            t_sin_low *= 1.0 / twilight_angle;
            glm_vec3_lerp(day, twilight, t_sin_low, sun_color);
        } else {
            glm_vec3_copy(day, sun_color);
        }

        /* increase just before sunrise/after sunset*/
        float intensity = t_sin + 0.07;
        if (intensity < 0) {
            intensity = 0;
        }

        color[i].r = sun_color[0];
        color[i].g = sun_color[1];
        color[i].b = sun_color[2];
        light_intensity[i].value = intensity;
    }
}

static
void AmbientLightControllerTimeOfDay(ecs_iter_t *it) {
    EcsTimeOfDay *tod = ecs_field(it, EcsTimeOfDay, 0);
    EcsCanvas *canvas = ecs_field(it, EcsCanvas, 1);

    static vec3 ambient_day = {0.03, 0.06, 0.09};
    static vec3 ambient_night = {0.001, 0.008, 0.016};
    static vec3 ambient_twilight = {0.01, 0.017, 0.02};
    static float twilight_zone = 0.2;

    for (int i = 0; i < it->count; i ++) {
        float t_sin = get_sun_height(tod[i].t);
        t_sin = (t_sin + 1.0) / 2;

        float t_twilight = glm_max(0.0, twilight_zone - fabs(t_sin - 0.5));
        t_twilight *= (1.0 / twilight_zone);

        vec3 ambient_color;
        glm_vec3_lerp(ambient_night, ambient_day, t_sin, ambient_color);
        glm_vec3_lerp(ambient_color, ambient_twilight, t_twilight, ambient_color);
        canvas[i].ambient_light.r = ambient_color[0];
        canvas[i].ambient_light.g = ambient_color[1];
        canvas[i].ambient_light.b = ambient_color[2];
    }
}

void FlecsGameLightControllerImport(ecs_world_t *world) {
    ECS_SYSTEM(world, LightControllerSyncPosition, EcsOnUpdate,
        [out]    flecs.components.graphics.DirectionalLight, 
        [in]     flecs.components.transform.Position3);

    ECS_SYSTEM(world, LightControllerSyncRotation, EcsOnUpdate,
        [out]    flecs.components.graphics.DirectionalLight, 
        [in]     flecs.components.transform.Position3,
        [in]     flecs.components.transform.Rotation3);

    ECS_SYSTEM(world, LightControllerSyncIntensity, EcsOnUpdate,
        [out]    flecs.components.graphics.DirectionalLight, 
        [in]     flecs.components.graphics.LightIntensity);

    ECS_SYSTEM(world, LightControllerSyncColor, EcsOnUpdate,
        [out]    flecs.components.graphics.DirectionalLight, 
        [in]     flecs.components.graphics.Rgb);

    ECS_SYSTEM(world, TimeOfDayUpdate, EcsOnUpdate,
        [inout]   TimeOfDay($));

    ECS_SYSTEM(world, LightControllerTimeOfDay, EcsOnUpdate,
        [in]      TimeOfDay($), 
        [out]     flecs.components.transform.Rotation3,
        [out]     flecs.components.graphics.Rgb,
        [out]     flecs.components.graphics.LightIntensity,
        [none]    flecs.components.graphics.Sun);

    ECS_SYSTEM(world, AmbientLightControllerTimeOfDay, EcsOnUpdate,
        [in]      TimeOfDay($), 
        [out]     flecs.components.gui.Canvas);

    ecs_add_pair(world, EcsSun, EcsWith, ecs_id(EcsRotation3));
    ecs_add_pair(world, EcsSun, EcsWith, ecs_id(EcsDirectionalLight));
    ecs_add_pair(world, EcsSun, EcsWith, ecs_id(EcsRgb));
    ecs_add_pair(world, EcsSun, EcsWith, ecs_id(EcsLightIntensity));
}

