#define FLECS_COMPONENTS_INPUT_IMPL

#include "flecs_components_input.h"
#include <string.h>

ECS_COMPONENT_DECLARE(EcsEventListener);
ECS_COMPONENT_DECLARE(EcsClassList);

static
ECS_MOVE(EcsEventListener, dst, src, {
    ecs_os_memcpy_t(dst, src, EcsEventListener);
    ecs_os_zeromem(src);
})

static
ECS_DTOR(EcsEventListener, ptr, {
    if (ptr->ctx && ptr->ctx_free) {
        ptr->ctx_free(ptr->ctx);
    }
    if (ptr->binding_ctx && ptr->binding_ctx_free) {
        ptr->binding_ctx_free(ptr->binding_ctx);
    }
})

static
ECS_CTOR(EcsClassList, ptr, {
    ecs_vec_init_t(NULL, &ptr->classes, ecs_entity_t, 0);
    ecs_vec_init_t(NULL, &ptr->added_by, ecs_classlist_added_by_t, 0);
})

static
ECS_MOVE(EcsClassList, dst, src, {
    ecs_vec_fini_t(NULL, &dst->classes, ecs_entity_t);
    ecs_vec_fini_t(NULL, &dst->added_by, ecs_classlist_added_by_t);
    ecs_os_memcpy_t(dst, src, EcsClassList);
    ecs_os_zeromem(src);
    ecs_vec_init_t(NULL, &src->classes, ecs_entity_t, 0);
    ecs_vec_init_t(NULL, &src->added_by, ecs_classlist_added_by_t, 0);
})

static
ECS_DTOR(EcsClassList, ptr, {
    ecs_vec_fini_t(NULL, &ptr->classes, ecs_entity_t);
    ecs_vec_fini_t(NULL, &ptr->added_by, ecs_classlist_added_by_t);
})

static
int32_t flecs_classlist_added_by_insert(
    ecs_classlist_added_by_t *arr,
    int32_t count,
    ecs_id_t added,
    ecs_entity_t by)
{
    for (int32_t i = 0; i < count; i ++) {
        if (arr[i].added == added) {
            arr[i].by = by;
            return count;
        }
        if (arr[i].added > added) {
            ecs_assert(count < ECS_INPUT_MAX_CLASS_COUNT, 
                ECS_INVALID_OPERATION, NULL);
            ecs_os_memmove_n(
                &arr[i + 1], &arr[i], ecs_classlist_added_by_t, count - i);
            arr[i].added = added;
            arr[i].by = by;
            return count + 1;
        }
    }

    ecs_assert(count < ECS_INPUT_MAX_CLASS_COUNT, 
        ECS_INVALID_OPERATION, NULL);
    arr[count].added = added;
    arr[count].by = by;
    return count + 1;
}

static 
void ecs_on_set(EcsClassList)(ecs_iter_t *it) {
    ecs_world_t *world = it->world;
    EcsClassList *class_list = ecs_field(it, EcsClassList, 1);
    ecs_classlist_added_by_t local_added_by[64];

    for (int32_t i = 0; i < it->count; i ++) {
        ecs_entity_t tgt = it->entities[i];
        ecs_vec_t *classes = &class_list[i].classes;
        ecs_vec_t *added_by = &class_list[i].added_by;
        int32_t class_count = ecs_vec_count(classes);
        ecs_entity_t *arr = ecs_vec_first(classes);
        int32_t added_by_count = 0;

        for (int32_t c = 0; c < class_count; c ++) {
            ecs_entity_t clss = arr[c];
            ecs_table_t *table = ecs_get_table(world, clss);
            const ecs_type_t *type = ecs_table_get_type(table);

            for (int32_t t = 0; t < type->count; t ++) {
                ecs_entity_t e = type->array[t];
                if (ECS_IS_PAIR(e)) {
                    e = ecs_pair_first(world, e);
                }

                if (ecs_has_pair(world, e, EcsOnInstantiate, EcsDontInherit)) {
                    continue;
                }

                added_by_count = flecs_classlist_added_by_insert(
                    local_added_by, added_by_count, type->array[t], clss);
            }
        }

        int32_t t_new = 0;
        ecs_classlist_added_by_t *ids_old = ecs_vec_first(added_by);
        for (int32_t t_old = 0; t_old < ecs_vec_count(added_by);) {
            ecs_id_t old_id = ids_old[t_old].added;
            ecs_id_t new_id = local_added_by[t_new].added;

            if (old_id < new_id) {
                ecs_remove_id(world, tgt, old_id);
            }

            t_old += old_id <= new_id;
            if (t_new < added_by_count) {
                t_new += new_id <= old_id;
            }
        }

        /* Copy new added ids to classlist */
        ecs_vec_set_count_t(NULL, added_by, 
            ecs_classlist_added_by_t, added_by_count);
        ecs_os_memcpy_n(ecs_vec_first(added_by), local_added_by, 
            ecs_classlist_added_by_t, added_by_count);

        /* Actually add the ids/set the components */
        for (int32_t t = 0; t < added_by_count; t ++) {
            ecs_id_t id = local_added_by[t].added;
            ecs_id_t by = local_added_by[t].by;

            const ecs_type_info_t *ti = ecs_get_type_info(world, id);
            if (!ti) {
                /* Not a component, just add the id */
                ecs_add_id(world, tgt, id);
            } else {
                /* Component, copy value */
                void *dst = ecs_ensure_id(world, tgt, id);
                void *src = ecs_ensure_id(world, by, id);
                if (ti->hooks.copy) {
                    ti->hooks.copy(dst, src, 1, ti);
                } else {
                    ecs_os_memcpy(dst, src, ti->size);
                }
            }
        }
    }
}

void FlecsComponentsInputImport(
    ecs_world_t *world)
{
    ECS_MODULE(world, FlecsComponentsInput);
    ECS_IMPORT(world, FlecsComponentsGraphics);

    ecs_set_name_prefix(world, "ecs");

    ECS_META_COMPONENT(world, ecs_key_state_t);
    ECS_META_COMPONENT(world, ecs_mouse_coord_t);
    ECS_META_COMPONENT(world, ecs_mouse_state_t);
    ECS_META_COMPONENT(world, ecs_classlist_added_by_t);

    ecs_set_name_prefix(world, "Ecs");

    ECS_META_COMPONENT(world, EcsInput);
    ECS_META_COMPONENT(world, EcsInputState);
    ECS_COMPONENT_DEFINE(world, EcsEventListener);
    ECS_COMPONENT_DEFINE(world, EcsClassList);

    ecs_add_pair(world, ecs_id(EcsEventListener), EcsWith, 
        ecs_id(EcsInputState));

    ecs_set_hooks(world, EcsInputState, {
        .ctor = flecs_default_ctor
    });

    ecs_set_hooks(world, EcsEventListener, {
        .ctor = flecs_default_ctor,
        .move = ecs_move(EcsEventListener),
        .dtor = ecs_dtor(EcsEventListener)
    });

    ecs_set_hooks(world, EcsClassList, {
        .ctor = ecs_ctor(EcsClassList),
        .move = ecs_move(EcsClassList),
        .dtor = ecs_dtor(EcsClassList),
        .on_set = ecs_on_set(EcsClassList)
    });

    ecs_entity_t classes_vec = ecs_vector(world, {
        .entity = ecs_entity(world, {
            .parent = ecs_id(FlecsComponentsInput),
        }),
        .type = ecs_id(ecs_entity_t)
    });

    ecs_entity_t added_by_vec = ecs_vector(world, {
        .entity = ecs_entity(world, {
            .parent = ecs_id(FlecsComponentsInput),
        }),
        .type = ecs_id(ecs_classlist_added_by_t)
    });

    ecs_struct(world, {
        .entity = ecs_id(EcsClassList),
        .members = {
            { "classes", classes_vec },
            { "added_by", added_by_vec }
        }
    });
}

