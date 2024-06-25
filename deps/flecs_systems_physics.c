#define FLECS_SYSTEMS_PHYSICS_IMPL
#include "flecs_systems_physics.h"

ECS_CTOR(EcsSpatialQuery, ptr, {
    ptr->query = NULL;
})

ECS_MOVE(EcsSpatialQuery, dst, src, {
    if (dst->query) {
        ecs_squery_free(dst->query);
    }

    dst->query = src->query;
    src->query = NULL;
})

ECS_DTOR(EcsSpatialQuery, ptr, {
    if (ptr->query) {
        ecs_squery_free(ptr->query);
    }
})

ECS_CTOR(EcsSpatialQueryResult, ptr, {
    ptr->results = (ecs_vec_t){0};
})

ECS_MOVE(EcsSpatialQueryResult, dst, src, {
    if (dst->results.array) {
        ecs_vec_fini_t(NULL, &dst->results, ecs_oct_entity_t);
    }

    dst->results = src->results;
    src->results = (ecs_vec_t){0};
})

ECS_DTOR(EcsSpatialQueryResult, ptr, {
    if (ptr->results.array) {
        ecs_vec_fini_t(NULL, &ptr->results, ecs_oct_entity_t);
    }
})

static
void EcsMove2(ecs_iter_t *it) {
    EcsPosition2 *p = ecs_field(it, EcsPosition2, 0);
    EcsVelocity2 *v = ecs_field(it, EcsVelocity2, 1);

    int i;
    for (i = 0; i < it->count; i ++) {
        p[i].x += v[i].x * it->delta_time;
        p[i].y += v[i].y * it->delta_time;
    }
}

static
void EcsMove3(ecs_iter_t *it) {
    EcsPosition3 *p = ecs_field(it, EcsPosition3, 0);
    EcsVelocity3 *v = ecs_field(it, EcsVelocity3, 1);

    int i;
    for (i = 0; i < it->count; i ++) {
        p[i].x += v[i].x * it->delta_time;
        p[i].y += v[i].y * it->delta_time;
        p[i].z += v[i].z * it->delta_time;
    }
}

static
void EcsRotate2(ecs_iter_t *it) {
    EcsRotation2 *r = ecs_field(it, EcsRotation2, 0);
    EcsAngularSpeed *a = ecs_field(it, EcsAngularSpeed, 1);

    int i;
    for (i = 0; i < it->count; i ++) {
        r[i].angle += a[i].value * it->delta_time;
    }
}

static
void EcsRotate3(ecs_iter_t *it) {
    EcsRotation3 *r = ecs_field(it, EcsRotation3, 0);
    EcsAngularVelocity *a = ecs_field(it, EcsAngularVelocity, 1);

    int i;
    for (i = 0; i < it->count; i ++) {
        r[i].x += a[i].x * it->delta_time;
        r[i].y += a[i].y * it->delta_time;
        r[i].z += a[i].z * it->delta_time;
    }
}

static
void EcsAddBoxCollider(ecs_iter_t *it) {
    EcsBox *box = ecs_field(it, EcsBox, 1);
    ecs_entity_t C = ecs_field_id(it, 0);
    ecs_entity_t B = ecs_field_id(it, 1);

    int i;
    if (ecs_field_is_self(it, 2)) {
        for (i = 0; i < it->count; i ++) {
            ecs_entity_t pair = ecs_pair(C, B);
            EcsBox *collider = ecs_ensure_id(
                it->world, it->entities[i], pair);
            ecs_os_memcpy_t(collider, &box[i], EcsBox);
        }
    } else {
        for (i = 0; i < it->count; i ++) {
            ecs_entity_t pair = ecs_pair(C, B);
            EcsBox *collider = ecs_ensure_id(
                it->world, it->entities[i], pair);
            ecs_os_memcpy_t(collider, box, EcsBox);
        }
    }
}

static
void EcsOnSetSpatialQuery(ecs_iter_t *it) {
    EcsSpatialQuery *q = ecs_field(it, EcsSpatialQuery, 0);
    ecs_id_t id = ecs_field_id(it, 0);
    ecs_id_t filter = ecs_pair_second(it->world, id);

    for (int i = 0; i < it->count; i ++) {
        q[i].query = ecs_squery_new(it->world, filter, q[i].center, q[i].size);
        if (!q[i].query) {
            char *filter_str = ecs_id_str(it->world, filter);
            ecs_err("failed to create query for filter '%s'", filter_str);
            ecs_os_free(filter_str);
        }
    }
}

static
void EcsUpdateSpatialQuery(ecs_iter_t *it) {
    EcsSpatialQuery *q = ecs_field(it, EcsSpatialQuery, 0);

    int i;
    for (i = 0; i < it->count; i ++) {
        if (!q->query) {
            char *filter_str = ecs_id_str(it->world, ecs_field_id(it, 1));
            ecs_err("missing spatial query for '%s'", filter_str);
            ecs_os_free(filter_str);
        }

        ecs_squery_update(q->query);
    }
}

void FlecsSystemsPhysicsImport(
    ecs_world_t *world)
{
    ECS_MODULE(world, FlecsSystemsPhysics);
    ECS_IMPORT(world, FlecsComponentsTransform);
    ECS_IMPORT(world, FlecsComponentsPhysics);
    ECS_IMPORT(world, FlecsComponentsGeometry);

    ecs_set_name_prefix(world, "Ecs");

    ECS_COMPONENT_DEFINE(world, EcsSpatialQuery);
    ECS_COMPONENT_DEFINE(world, EcsSpatialQueryResult);

    ecs_struct(world, {
        .entity = ecs_id(EcsSpatialQuery),
        .members = {
            {"center", ecs_id(vec3)},
            {"size", ecs_id(ecs_f32_t)}
        }
    });

    ecs_set_hooks(world, EcsSpatialQuery, {
        .ctor = ecs_ctor(EcsSpatialQuery),
        .dtor = ecs_dtor(EcsSpatialQuery),
        .move = ecs_move(EcsSpatialQuery)
    });

    ecs_add_pair(world, ecs_id(EcsSpatialQuery), EcsOnInstantiate, EcsInherit);

    ecs_set_hooks(world, EcsSpatialQueryResult, {
        .ctor = ecs_ctor(EcsSpatialQueryResult),
        .dtor = ecs_dtor(EcsSpatialQueryResult),
        .move = ecs_move(EcsSpatialQueryResult)
    });    

    ECS_SYSTEM(world, EcsMove2, EcsOnUpdate, 
        [inout] flecs.components.transform.Position2,
        [in]    flecs.components.physics.Velocity2);

    ECS_SYSTEM(world, EcsMove3, EcsOnUpdate, 
        [inout] flecs.components.transform.Position3,
        [in]    flecs.components.physics.Velocity3);

    ECS_SYSTEM(world, EcsRotate2, EcsOnUpdate, 
        [inout] flecs.components.transform.Rotation2,
        [in]    flecs.components.physics.AngularSpeed);

    ECS_SYSTEM(world, EcsRotate3, EcsOnUpdate, 
        [inout] flecs.components.transform.Rotation3,
        [in]    flecs.components.physics.AngularVelocity);

    ECS_SYSTEM(world, EcsAddBoxCollider, EcsPostLoad, 
        flecs.components.physics.Collider,
        flecs.components.geometry.Box,
        !(flecs.components.physics.Collider, flecs.components.geometry.Box));

    ECS_OBSERVER(world, EcsOnSetSpatialQuery, EcsOnSet,
        SpatialQuery(self, *), ?Prefab);

    ECS_SYSTEM(world, EcsUpdateSpatialQuery, EcsPreUpdate, 
        SpatialQuery(self, *), ?Prefab);

    ecs_system(world, { .entity = EcsMove2,   .query.flags = EcsQueryIsInstanced });
    ecs_system(world, { .entity = EcsMove3,   .query.flags = EcsQueryIsInstanced });
    ecs_system(world, { .entity = EcsRotate3, .query.flags = EcsQueryIsInstanced });

    ecs_add_pair(world, ecs_id(EcsVelocity2), 
        EcsWith, ecs_id(EcsPosition2));
    ecs_add_pair(world, ecs_id(EcsVelocity3), 
        EcsWith, ecs_id(EcsPosition3));
    ecs_add_pair(world, ecs_id(EcsRotation3), 
        EcsWith, ecs_id(EcsPosition3));
    ecs_add_pair(world, ecs_id(EcsAngularVelocity), 
        EcsWith, ecs_id(EcsRotation3));
}


#define MAX_PER_OCTANT (8)

typedef struct cube_t {
    struct cube_t *parent;
    struct cube_t *nodes[8];
    ecs_vec_t entities;
    int32_t id;
    bool is_leaf;
} cube_t;

struct ecs_octree_t {
    ecs_sparse_t cubes;
    ecs_vec_t free_cubes;
    cube_t root;
    vec3 center;
    float size;
    int32_t count;
};

static
void cube_split(
    ecs_octree_t *ot,
    cube_t *cube,
    vec3 center,
    float size);

static
cube_t *new_cube(
    ecs_octree_t *ot,
    cube_t *parent)
{
    cube_t *result = NULL;
    if (!ecs_vec_count(&ot->free_cubes)) {
        result = ecs_sparse_add_t(&ot->cubes, cube_t);
        result->id = (int32_t)ecs_sparse_last_id(&ot->cubes);
    } else {
        result = ecs_vec_last_t(&ot->free_cubes, cube_t*)[0];
        ecs_vec_remove_last(&ot->free_cubes);
    }
    
    result->parent = parent;
    result->is_leaf = true;

    return result;
}

static
bool is_inside_dim(
    float center,
    float size,
    float e_pos,
    float e_size)
{
    bool 
    result =  e_pos - e_size >= center - size;
    result &= e_pos + e_size <= center + size;
    return result;
}

static
bool is_inside(
    vec3 center,
    float size,
    vec3 e_pos,
    vec3 e_size)
{
    bool 
    result =  is_inside_dim(center[0], size, e_pos[0], e_size[0]);
    result &= is_inside_dim(center[1], size, e_pos[1], e_size[1]);
    result &= is_inside_dim(center[2], size, e_pos[2], e_size[2]);
    return result;
}

/* Returns 0 if no overlap, 2 if contains */
static
int overlap_dim(
    float center,
    float size,
    float pos,
    float range)
{
    float left = center - size;
    float right = center + size;
    float q_left = pos - range;
    float q_right = pos + range;

    bool left_outside = q_left < left;
    bool left_nomatch = q_left > right;

    bool right_outside = q_right > right;
    bool right_nomatch = q_right < left;

    return (!(left_nomatch | right_nomatch)) * (1 + (left_outside && right_outside));
}

/* Returns 0 if no overlap, 8 if contains, < 8 if overlaps */
static
int entity_overlaps(
    vec3 center,
    float size,
    vec3 e_pos,
    vec3 e_size)
{
    int result = overlap_dim(center[0], size, e_pos[0], e_size[0]);
    result *=    overlap_dim(center[1], size, e_pos[1], e_size[1]);
    result *=    overlap_dim(center[2], size, e_pos[2], e_size[2]);
    return result;
}

static
int cube_overlaps(
    vec3 center,
    float size,
    vec3 pos,
    float range)
{
    int result = overlap_dim(center[0], size, pos[0], range);
    result *=    overlap_dim(center[1], size, pos[1], range);
    result *=    overlap_dim(center[2], size, pos[2], range);
    return result;
}

/* Convenience macro for contains */
#define CONTAINS_CUBE (8)

static
int8_t get_side(
    float center,
    float coord)
{
    return coord > center;
}

static
int8_t next_cube_index(
    vec3 center,
    float size,
    vec3 pos)    
{
    int8_t left_right = get_side(center[0], pos[0]);
    int8_t top_bottom = get_side(center[1], pos[1]);
    int8_t front_back = get_side(center[2], pos[2]);
    center[0] += size * (left_right * 2 - 1);
    center[1] += size * (top_bottom * 2 - 1);
    center[2] += size * (front_back * 2 - 1);
    return left_right + top_bottom * 2 + front_back * 4;
}

static const vec3 cube_map[] = {
    {-1, -1, -1},
    { 1, -1, -1},
    {-1,  1, -1},
    { 1,  1, -1},
    {-1, -1,  1},
    { 1, -1,  1},
    {-1,  1,  1},
    { 1,  1,  1},
};

static
void get_cube_center(
    vec3 center,
    float size,
    int8_t index,
    vec3 result)
{
    result[0] = center[0] + cube_map[index][0] * size;
    result[1] = center[1] + cube_map[index][1] * size;
    result[2] = center[2] + cube_map[index][2] * size;
}

static
void cube_add_entity(
    cube_t *cube,
    ecs_oct_entity_t *ce)
{
    ecs_vec_init_if_t(&cube->entities, ecs_oct_entity_t);
    ecs_oct_entity_t *elem = ecs_vec_append_t(
        NULL, &cube->entities, ecs_oct_entity_t);
    *elem = *ce;
}

static
cube_t* cube_insert(
    ecs_octree_t *ot,
    ecs_oct_entity_t *ce,
    cube_t *cube,
    vec3 cube_center,
    float cube_size)
{
    /* create private variable that can be modified during iteration */
    vec3 center;
    glm_vec3_copy(cube_center, center);
    float size = cube_size / 2;
    cube_t *cur = cube;
    ecs_oct_entity_t e = *ce;

    if (!is_inside(center, size, ce->pos, ce->size)) {
        return NULL;
    }

    do {
        bool is_leaf = cur->is_leaf;

        /* If cube is a leaf and has space, insert */
        if (is_leaf && ecs_vec_count(&cur->entities) < MAX_PER_OCTANT) {
            break;
        }

        /* Find next cube */
        vec3 child_center;
        glm_vec3_copy(center, child_center);
        float child_size = size / 2;
        int8_t cube_i = next_cube_index(child_center, child_size, e.pos);
        
        /* If entity does not fit in child cube, insert into current */
        if (!is_inside(child_center, child_size, e.pos, e.size)) {
            break;
        }

        /* Entity should not be inserted in current node. Check if node is a
         * leaf. If it is, split it up */
        if (is_leaf) {
            cube_split(ot, cur, center, size * 2);
        }

        cube_t *next = cur->nodes[cube_i];
        if (!next) {
            next = new_cube(ot, cur);
            ecs_assert(next != cur, ECS_INTERNAL_ERROR, NULL);

            cur->nodes[cube_i] = next;
            cur = next;

            /* Node is guaranteed to be empty, so stop looking */
            break;
        } else {
            ecs_assert(next != cur, ECS_INTERNAL_ERROR, NULL);
            cur = next;
        }

        size = child_size;
        glm_vec3_copy(child_center, center);
    } while (1);

    cube_add_entity(cur, &e);

    return cur;
}

static
void cube_split(
    ecs_octree_t *ot,
    cube_t *cube,
    vec3 center,
    float size)
{
    int32_t i, count = ecs_vec_count(&cube->entities);

    /* This will force entities to be pushed to child nodes */
    cube->is_leaf = false; 

    for (i = 0; i < count; i ++) {
        ecs_oct_entity_t *entities = ecs_vec_first_t(&cube->entities, ecs_oct_entity_t);
        cube_t *new_cube = cube_insert(ot, &entities[i], cube, center, size);
        ecs_assert(new_cube != NULL, ECS_INTERNAL_ERROR, NULL);

        if (new_cube != cube) {
            ecs_vec_remove_t(&cube->entities, ecs_oct_entity_t, i);
            i --;
            count --;
            ecs_assert(count == ecs_vec_count(&cube->entities), ECS_INTERNAL_ERROR, NULL);
        } else {
            ecs_vec_remove_last(&cube->entities);
        }
    }

    ecs_assert(count == ecs_vec_count(&cube->entities), ECS_INTERNAL_ERROR, NULL);
}

static
void result_add_entity(
    ecs_vec_t *result,
    ecs_oct_entity_t *e)
{
    ecs_oct_entity_t *elem = ecs_vec_append_t(NULL, result, ecs_oct_entity_t);
    *elem = *e;
}

static
void cube_find_all(
    cube_t *cube,
    ecs_vec_t *result)
{
    ecs_vec_init_if_t(result, ecs_oct_entity_t);
    ecs_oct_entity_t *entities = ecs_vec_first_t(&cube->entities, ecs_oct_entity_t);
    int32_t i, count = ecs_vec_count(&cube->entities);
    for (i = 0; i < count; i ++) {
        result_add_entity(result, &entities[i]);
    }

    for (i = 0; i < 8; i ++) {
        cube_t *child = cube->nodes[i];
        if (!child) {
            continue;
        }

        cube_find_all(child, result);
    }
}

static
void cube_findn(
    cube_t *cube,
    vec3 center,
    float size,
    vec3 pos,
    float range,
    ecs_vec_t *result)
{
    size /= 2;

    ecs_vec_init_if_t(result, ecs_oct_entity_t);
    ecs_oct_entity_t *entities = ecs_vec_first_t(&cube->entities, ecs_oct_entity_t);
    int32_t i, count = ecs_vec_count(&cube->entities);

    for (i = 0; i < count; i ++) {
        ecs_oct_entity_t *e = &entities[i];
        if (entity_overlaps(pos, range, e->pos, e->size)) {
            result_add_entity(result, e);
        }
    }

    for (i = 0; i < 8; i ++) {
        cube_t *child = cube->nodes[i];
        if (!child) {
            continue;
        }

        vec3 child_center;
        get_cube_center(center, size, i, child_center);
        int overlap = cube_overlaps(child_center, size, pos, range);

        if (overlap == CONTAINS_CUBE) {
            cube_find_all(child, result);
        } else if (overlap) {
            cube_findn(child, child_center, size, pos, range, result);
        }
    }
}

ecs_octree_t* ecs_octree_new(
    vec3 center,
    float size)
{
    ecs_octree_t *result = ecs_os_calloc(sizeof(ecs_octree_t));
    glm_vec3_copy(center, result->center);
    result->size = size;
    ecs_sparse_init_t(&result->cubes, cube_t);
    return result;
}

void ecs_octree_clear(
    ecs_octree_t *ot)
{
    ecs_assert(ot != NULL, ECS_INVALID_PARAMETER, NULL);

    /* Keep existing cubes intact so that we can reuse them when the octree is
     * repopulated. This lets us keep the entity vectors, and should cause the
     * octree memory to stabilize eventually. */
    int32_t i, count = ecs_sparse_count(&ot->cubes);
    for (i = 0; i < count; i ++) {
        cube_t *cube = ecs_sparse_get_dense_t(&ot->cubes, cube_t, i);
        ecs_vec_clear(&cube->entities);
        ecs_os_memset_n(cube->nodes, 0, cube_t*, 8);

        if (cube->parent) {
            ecs_vec_init_if_t(&ot->free_cubes, cube_t*);
            cube_t **cptr = ecs_vec_append_t(NULL, &ot->free_cubes, cube_t*);
            *cptr = cube;
            cube->parent = NULL;
        }
    }

    /* Clear entities of root */
    ecs_vec_clear(&ot->root.entities);
    ecs_os_memset_n(ot->root.nodes, 0, cube_t*, 8);
    ot->count = 0;
}

void ecs_octree_free(
    ecs_octree_t *ot)
{
}

int32_t ecs_octree_insert(
    ecs_octree_t *ot,
    ecs_entity_t e,
    vec3 e_pos,
    vec3 e_size)
{
    ecs_assert(ot != NULL, ECS_INVALID_PARAMETER, NULL);

    ecs_oct_entity_t ce;
    ce.id = e;
    glm_vec3_copy(e_pos, ce.pos);
    glm_vec3_copy(e_size, ce.size);
    cube_t *cube = cube_insert(ot, &ce, &ot->root, ot->center, ot->size);
    if (cube) {
        ot->count ++;
        return cube->id;
    } else {
        return -1;
    }
}

void ecs_octree_findn(
    ecs_octree_t *ot,
    vec3 pos,
    float range,
    ecs_vec_t *result)
{
    ecs_assert(ot != NULL, ECS_INVALID_PARAMETER, NULL);

    ecs_vec_clear(result);
    cube_findn(&ot->root, ot->center, ot->size / 2, pos, range, result);
}

static
int cube_dump(
    cube_t *cube,
    vec3 center,
    float size)
{
    vec3 c;
    glm_vec3_copy(center, c);

    size /= 2;
    int i, count = 0;
    for (i = 0; i < 8; i ++) {
        if (cube->nodes[i]) {
            vec3 child_center;
            get_cube_center(c, size, i, child_center);
            count += cube_dump(cube->nodes[i], child_center, size);
        }
    }

    return ecs_vec_count(&cube->entities) + count;
}

int32_t ecs_octree_dump(
    ecs_octree_t *ot)
{
    ecs_assert(ot != NULL, ECS_INVALID_PARAMETER, NULL);
    int32_t ret = cube_dump(&ot->root, ot->center, ot->size / 2);
    printf("counted = %d, actual = %d\n", ret, ot->count);
    ecs_assert(ret == ot->count, ECS_INTERNAL_ERROR, NULL);
    return ret;
}


struct ecs_squery_t {
    ecs_query_t *q;
    ecs_octree_t *ot;
};

#define EXPR_PREFIX\
    "[in] flecs.components.transform.Position3,"\
    "[in] (flecs.components.physics.Collider, flecs.components.geometry.Box) || flecs.components.geometry.Box,"

ecs_squery_t* ecs_squery_new(
    ecs_world_t *world,
    ecs_id_t filter,
    vec3 center,
    float size)
{
    ecs_assert(world != NULL, ECS_INVALID_PARAMETER, NULL);
    ecs_assert(size > 0, ECS_INVALID_PARAMETER, NULL);

    ecs_squery_t *result = ecs_os_calloc(sizeof(ecs_squery_t));

    result->q = ecs_query(world, {
        .terms = {
            { ecs_id(EcsPosition3), .inout = EcsIn },
            { ecs_pair(EcsCollider, ecs_id(EcsBox)), .inout = EcsIn, .oper = EcsOr }, 
            { ecs_id(EcsBox) },
            { filter, .inout = EcsIn }
        },
        .flags = EcsQueryIsInstanced,
        .cache_kind = EcsQueryCacheAuto
    });

    result->ot = ecs_octree_new(center, size);

    ecs_assert(result->q != NULL, ECS_INTERNAL_ERROR, NULL);
    ecs_assert(result->ot != NULL, ECS_INTERNAL_ERROR, NULL);

    return result;
}

void ecs_squery_free(
    ecs_squery_t *sq)
{
    ecs_query_fini(sq->q);
    ecs_octree_free(sq->ot);
    ecs_os_free(sq);
}

void ecs_squery_update(
    ecs_squery_t *sq)
{
    ecs_assert(sq != NULL,     ECS_INVALID_PARAMETER, NULL);
    ecs_assert(sq->q != NULL,  ECS_INVALID_PARAMETER, NULL);
    ecs_assert(sq->ot != NULL, ECS_INVALID_PARAMETER, NULL);

    if (ecs_query_changed(sq->q)) {
        ecs_octree_clear(sq->ot);

        const ecs_world_t *world = ecs_get_world(sq->q);
        ecs_iter_t it = ecs_query_iter(world, sq->q);
        while (ecs_query_next(&it)) {
            EcsPosition3 *p = ecs_field(&it, EcsPosition3, 0);
            EcsBox *b = ecs_field(&it, EcsBox, 1);

            if (ecs_field_is_self(&it, 1)) {
                int i;
                for (i = 0; i < it.count; i ++) {
                    vec3 vp, vs;
                    vp[0] = p[i].x;
                    vp[1] = p[i].y;
                    vp[2] = p[i].z;

                    vs[0] = b[i].width;
                    vs[1] = b[i].height;
                    vs[2] = b[i].depth;

                    ecs_octree_insert(sq->ot, it.entities[i], vp, vs);
                }
            } else {
                int i;
                for (i = 0; i < it.count; i ++) {
                    vec3 vp, vs;
                    vp[0] = p[i].x;
                    vp[1] = p[i].y;
                    vp[2] = p[i].z;

                    vs[0] = b->width;
                    vs[1] = b->height;
                    vs[2] = b->depth;

                    ecs_octree_insert(sq->ot, it.entities[i], vp, vs);
                }
            }
        }
    }
}

void ecs_squery_findn(
    const ecs_squery_t *sq,
    vec3 position,
    float range,
    ecs_vec_t *result)
{
    ecs_assert(sq != NULL, ECS_INVALID_PARAMETER, NULL);
    ecs_assert(sq->q != NULL, ECS_INVALID_PARAMETER, NULL);
    ecs_assert(sq->ot != NULL, ECS_INVALID_PARAMETER, NULL);    

    ecs_octree_findn(sq->ot, position, range, result);
}

