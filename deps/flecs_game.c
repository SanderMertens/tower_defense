#define FLECS_GAME_IMPL

#include "flecs_game.h"

ECS_DECLARE(EcsWorldCell);
ECS_DECLARE(EcsCameraController);

void FlecsGameCameraControllerImport(ecs_world_t *world);
void FlecsGameWorldCellsImport(ecs_world_t *world);

void FlecsGameImport(ecs_world_t *world) {
    ECS_MODULE(world, FlecsGame);

    ECS_IMPORT(world, FlecsComponentsTransform);
    ECS_IMPORT(world, FlecsComponentsPhysics);
    ECS_IMPORT(world, FlecsComponentsGraphics);
    ECS_IMPORT(world, FlecsComponentsInput);
    ECS_IMPORT(world, FlecsSystemsPhysics);

    ecs_set_name_prefix(world, "Ecs");

    ECS_TAG_DEFINE(world, EcsCameraController);
    ECS_META_COMPONENT(world, EcsCameraAutoMove);
    ECS_META_COMPONENT(world, EcsWorldCellCoord);

    FlecsGameCameraControllerImport(world);
    FlecsGameWorldCellsImport(world);
}


ECS_DECLARE(EcsWorldCell);
ECS_DECLARE(EcsWorldCellRoot);
ECS_COMPONENT_DECLARE(WorldCells);
ECS_COMPONENT_DECLARE(WorldCellCache);

typedef struct ecs_world_quadrant_t {
    ecs_map_t cells;
} ecs_world_quadrant_t;

typedef struct WorldCells {
    ecs_world_quadrant_t quadrants[4];
} WorldCells;

typedef struct WorldCellCache {
    uint64_t cell_id;
    uint64_t old_cell_id;
    int8_t quadrant;
    int8_t old_quadrant;
} WorldCellCache;

static
void flecs_game_get_cell_id(
    WorldCellCache *cache,
    float xf, 
    float yf)
{
    int32_t x = xf;
    int64_t y = yf;

    uint8_t left = x < 0;
    uint8_t bottom = y < 0;

    x *= 1 - (2 * left);
    y *= 1 - (2 * bottom);

    x = x >> FLECS_GAME_WORLD_CELL_SHIFT;
    y = y >> FLECS_GAME_WORLD_CELL_SHIFT;

    cache->quadrant = left + bottom * 2;
    cache->cell_id = x + (y << 32);
}

static
ecs_entity_t flecs_game_get_cell(
    ecs_world_t *world,
    WorldCells *wcells,
    const WorldCellCache *wcache)
{
    int8_t quadrant = wcache->quadrant;
    uint64_t cell_id = wcache->cell_id;
    ecs_entity_t *cell_ptr = ecs_map_ensure(&wcells->quadrants[quadrant].cells, 
        ecs_entity_t, cell_id);
    ecs_entity_t cell = *cell_ptr;
    if (!cell) {
        cell = *cell_ptr = ecs_new(world, EcsWorldCell);
        ecs_add_pair(world, cell, EcsChildOf, EcsWorldCellRoot);

        // Decode cell coordinates from spatial hash
        int32_t left = (int32_t)cell_id;
        int32_t bottom = (int32_t)(cell_id >> 32);
        int32_t half_size = (1 << FLECS_GAME_WORLD_CELL_SHIFT) / 2;
        bottom = bottom << FLECS_GAME_WORLD_CELL_SHIFT;
        left = left << FLECS_GAME_WORLD_CELL_SHIFT;
        int32_t x = left + half_size;
        int32_t y = bottom + half_size;
        if (wcache->quadrant & 1) {
            x *= -1;
        }
        if (wcache->quadrant & 2) {
            y *= -1;
        }

        ecs_set(world, cell, EcsWorldCellCoord, {
            .x = x,
            .y = y,
            .size = 1 << FLECS_GAME_WORLD_CELL_SHIFT
        });
    }
    return cell;
}

static
void AddWorldCellCache(ecs_iter_t *it) {
    ecs_world_t *world = it->world;

    for (int i = 0; i < it->count; i ++) {
        ecs_set(world, it->entities[i], WorldCellCache, { 
            .cell_id = 0, .old_cell_id = -1
        });
    }    
}

static
void FindWorldCell(ecs_iter_t *it) {
    while (ecs_query_next_table(it)) {
        if (!ecs_query_changed(NULL, it)) {
            continue;
        }

        ecs_query_populate(it);

        EcsPosition3 *pos = ecs_field(it, EcsPosition3, 1);
        WorldCellCache *wcache = ecs_field(it, WorldCellCache, 2);

        for (int i = 0; i < it->count; i ++) {
            flecs_game_get_cell_id(&wcache[i], pos[i].x, pos[i].z);
        }
    }
}

static
void SetWorldCell(ecs_iter_t *it) {
    while (ecs_query_next_table(it)) {
        if (!ecs_query_changed(NULL, it)) {
            continue;
        }

        ecs_query_populate(it);

        ecs_world_t *world = it->world;
        WorldCellCache *wcache = ecs_field(it, WorldCellCache, 1);
        WorldCells *wcells = ecs_field(it, WorldCells, 2);

        for (int i = 0; i < it->count; i ++) {
            WorldCellCache *cur = &wcache[i];

            if (cur->cell_id != cur->old_cell_id || cur->quadrant != cur->old_quadrant) {
                ecs_entity_t cell = flecs_game_get_cell(world, wcells, cur);
                ecs_add_pair(world, it->entities[i], ecs_id(EcsWorldCell), cell);
            }
        }
    }
}

static
void ResetWorldCellCache(ecs_iter_t *it) {
    while (ecs_query_next_table(it)) {
        if (!ecs_query_changed(NULL, it)) {
            continue;
        }

        ecs_query_populate(it);

        WorldCellCache *wcache = ecs_field(it, WorldCellCache, 1);
        bool changed = false;

        for (int i = 0; i < it->count; i ++) {
            WorldCellCache *cur = &wcache[i];
            if (cur->old_cell_id != cur->cell_id || cur->old_quadrant != cur->quadrant) {
                cur->old_cell_id = cur->cell_id;
                cur->old_quadrant = cur->quadrant;
                changed = true;
            }
        }

        if (!changed) {
            ecs_query_skip(it);
        }
    }
}

void FlecsGameWorldCellsImport(ecs_world_t *world) {

    ECS_COMPONENT_DEFINE(world, WorldCellCache);
    ECS_COMPONENT_DEFINE(world, WorldCells);
    ECS_ENTITY_DEFINE(world, EcsWorldCell, Tag, Exclusive);

    ecs_set_hooks(world, WorldCells, {
        .ctor = ecs_default_ctor
    });

    EcsWorldCellRoot = ecs_entity(world, {
        .name = "::game.worldcells",
        .root_sep = "::"
    });

    ECS_SYSTEM(world, AddWorldCellCache, EcsOnLoad,
        [none] flecs.components.transform.Position3(self),
        [out]  !flecs.game.WorldCellCache(self),
        [out]  !flecs.game.WorldCell(self),
        [none] !flecs.components.transform.Position3(up(ChildOf)));

    ecs_system(world, {
        .entity = ecs_entity(world, {
            .name = "FindWorldCell",
            .add = { ecs_dependson(EcsOnValidate) }
        }),
        .query = {
            .filter.terms = {{
                .id = ecs_id(EcsPosition3),
                .inout = EcsIn,
                .src.flags = EcsSelf
            }, {
                .id = ecs_id(WorldCellCache),
                .inout = EcsOut,
                .src.flags = EcsSelf
            }}
        },
        .run = FindWorldCell
    });

    ecs_system(world, {
        .entity = ecs_entity(world, {
            .name = "SetWorldCell",
            .add = { ecs_dependson(EcsOnValidate) }
        }),
        .query = {
            .filter.terms = {{
                .id = ecs_id(WorldCellCache),
                .inout = EcsIn,
                .src.flags = EcsSelf
            }, {
                .id = ecs_id(WorldCells),
                .inout = EcsIn,
                .src.flags = EcsSelf,
                .src.id = ecs_id(WorldCells)
            }, {
                .id = ecs_pair(EcsWorldCell, EcsWildcard),
                .inout = EcsOut,
                .src.id = 0,
                .src.flags = EcsIsEntity
            }}
        },
        .run = SetWorldCell
    });

    ecs_system(world, {
        .entity = ecs_entity(world, {
            .name = "ResetWorldCellCache",
            .add = { ecs_dependson(EcsOnValidate) }
        }),
        .query = {
            .filter.terms = {{
                .id = ecs_id(WorldCellCache),
                .inout = EcsInOut,
                .src.flags = EcsSelf
            }}
        },
        .run = ResetWorldCellCache
    });

    WorldCells *wcells = ecs_singleton_get_mut(world, WorldCells);
    ecs_map_init(&wcells->quadrants[0].cells, ecs_entity_t, NULL, 1);
    ecs_map_init(&wcells->quadrants[1].cells, ecs_entity_t, NULL, 1);
    ecs_map_init(&wcells->quadrants[2].cells, ecs_entity_t, NULL, 1);
    ecs_map_init(&wcells->quadrants[3].cells, ecs_entity_t, NULL, 1);
}


#define CAMERA_DECELERATION 100.0
#define CAMERA_ANGULAR_DECELERATION 5.0

static const float CameraDeceleration = CAMERA_DECELERATION;
static const float CameraAcceleration = 50.0 + CAMERA_DECELERATION;
static const float CameraAngularDeceleration = CAMERA_ANGULAR_DECELERATION;
static const float CameraAngularAcceleration = 2.5 + CAMERA_ANGULAR_DECELERATION;
static const float CameraMaxSpeed = 50.0;

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

static
void CameraAutoMove(ecs_iter_t *it) {
    EcsVelocity3 *v = ecs_field(it, EcsVelocity3, 1);
    EcsCameraAutoMove *m = ecs_field(it, EcsCameraAutoMove, 2);

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

