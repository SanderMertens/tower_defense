#include "flecs_systems_transform.h"

void EcsAddTransform3(ecs_iter_t *it) {
    ecs_world_t *world = it->world;
    ecs_entity_t comp = ecs_field_id(it, 1);

    int i;
    for (i = 0; i < it->count; i ++) {
        ecs_add_id(world, it->entities[i], comp);
    }
}

void EcsApplyTransform3(ecs_iter_t *it) {
    // Iterate, only table field is populated to reduce overhead
    while (ecs_query_next_table(it)) {
        // Table has changed, populate fields so that we can run the system
        // code as usual
        int result;
        bool changed = false;
        while ((result = ecs_query_populate(it, true)) != EcsIterNext) {
            EcsTransform3 *m = ecs_field(it, EcsTransform3, 1);
            EcsTransform3 *m_parent = ecs_field(it, EcsTransform3, 2);
            EcsPosition3 *p = ecs_field(it, EcsPosition3, 3);
            EcsRotation3 *r = ecs_field(it, EcsRotation3, 4);
            EcsScale3 *s = ecs_field(it, EcsScale3, 5);
            changed = true;
            int i;

            if (!m_parent) {
                if (ecs_field_is_self(it, 3)) {
                    for (i = 0; i < it->count; i ++) {
                        glm_translate_make(m[i].value, *(vec3*)&p[i]);
                    }
                } else {
                    for (i = 0; i < it->count; i ++) {
                        glm_translate_make(m[i].value, *(vec3*)p);
                    }
                }
            } else {
                if (ecs_field_is_self(it, 3)) {
                    for (i = 0; i < it->count; i ++) {
                        glm_translate_to(m_parent[0].value, *(vec3*)&p[i], m[i].value);
                    }
                } else {
                    for (i = 0; i < it->count; i ++) {
                        glm_translate_to(m_parent[0].value, *(vec3*)p, m[i].value);
                    }
                }
            }

            if (r) {
                if (ecs_field_is_self(it, 4)) {
                    for (i = 0; i < it->count; i ++) {
                        glm_rotate(m[i].value, r[i].x, (vec3){1.0, 0.0, 0.0});
                        glm_rotate(m[i].value, r[i].y, (vec3){0.0, 1.0, 0.0});
                        glm_rotate(m[i].value, r[i].z, (vec3){0.0, 0.0, 1.0});
                    }
                } else {
                    for (i = 0; i < it->count; i ++) {
                        glm_rotate(m[i].value, r->x, (vec3){1.0, 0.0, 0.0});
                        glm_rotate(m[i].value, r->y, (vec3){0.0, 1.0, 0.0});
                        glm_rotate(m[i].value, r->z, (vec3){0.0, 0.0, 1.0});
                    }
                }
            }

            if (s) {
                for (i = 0; i < it->count; i ++) {
                    glm_scale(m[i].value, *(vec3*)&s[i]);
                }
            }

            if (result == EcsIterNextYield) {
                break;
            }
        }
        if (!changed) {
            ecs_query_skip(it);
        }
    }
}

void FlecsSystemsTransformImport(
    ecs_world_t *world)
{
    ECS_MODULE(world, FlecsSystemsTransform);
    ECS_IMPORT(world, FlecsComponentsTransform);

    ecs_set_name_prefix(world, "Ecs");

    ecs_add_pair(world, ecs_id(EcsPosition3), EcsWith, ecs_id(EcsTransform3));
    ecs_add_pair(world, ecs_id(EcsRotation3), EcsWith, ecs_id(EcsTransform3));
    ecs_add_pair(world, ecs_id(EcsScale3), EcsWith, ecs_id(EcsTransform3));
    ecs_add_id(world, ecs_id(EcsTransform3), EcsAlwaysOverride);

    ecs_system(world, {
        .entity = ecs_entity(world, { 
            .name = "EcsApplyTransform3",
            .add = { ecs_dependson(EcsOnValidate) }
        }),
        .query = {
            .filter = {
                .terms = {{ 
                    .id = ecs_id(EcsTransform3),
                    .inout = EcsOut,
                    .src.flags = EcsSelf
                },
                {
                    .id = ecs_id(EcsTransform3), 
                    .inout = EcsIn,
                    .oper = EcsOptional,
                    .src.flags = EcsParent | EcsCascade
                },
                {
                    .id = ecs_id(EcsPosition3),
                    .inout = EcsIn
                },
                {
                    .id = ecs_id(EcsRotation3),
                    .inout = EcsIn,
                    .oper = EcsOptional
                },
                {
                    .id = ecs_id(EcsScale3),
                    .inout = EcsIn,
                    .oper = EcsOptional
                }},
                .instanced = true
            },
        },
        .run = EcsApplyTransform3
    });
}

