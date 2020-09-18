#include <tower_defense.h>
#include <vector>

using namespace flecs::components;

using Position = transform::Position3;
using Rotation = transform::Rotation3;
using Velocity = physics::Velocity3;
using SpatialQuery = flecs::systems::physics::SpatialQuery;
using SpatialQueryResult = flecs::systems::physics::SpatialQueryResult;
using Color = geometry::Color;
using Box = geometry::Box;

#define ECS_PI_2 ((float)(GLM_PI * 2))

// Game constants
static const float CameraAcceleration = 0.2;
static const float CameraDeceleration = 0.1;
static const float CameraMaxSpeed = 0.05;
static const float CameraDistance = 9;
static const float CameraHeight = 6.0;
static const float CameraLookatRadius = 2.0;

static const float EnemySize = 0.3;
static const float EnemySpeed = 1.5;
static const float EnemySpawnInterval = 1.0;

static const float TurretRotateSpeed = 2.0;
static const float TurretFireInterval = 0.1;
static const float TurretRange = 3.0;
static const float TurretCannonOffset = 0.1;

static const float BulletSize = 0.06;
static const float BulletSpeed = 10.0;
static const float BulletLifespan = 0.5;

static const float TileSize = 1.5;
static const float TileHeight = 0.4;
static const float PathHeight = 0.1;
static const float TileSpacing = 0.0;
static const int TileCountX = 10;
static const int TileCountZ = 10;

// Direction vector. During pathfinding enemies will cycle through this vector
// to find the next direction to turn to.
static const transform::Position2 dir[] = {
    {-1, 0},
    {0, -1},
    {1, 0},
    {0, 1},
};

// Grid wrapper around a vector which translates an (x, y) into a vector index
template <typename T>
class grid {
public:
    grid(int width, int height)
        : m_width(width)
        , m_height(height) 
    { 
        for (int x = 0; x < width; x ++) {
            for (int y = 0; y < height; y ++) {
                m_values.push_back(T());
            }
        }
    }

    void set(int32_t x, int32_t y, T value) {
        m_values[y * m_width + x] = value;
    }

    const T operator()(int32_t x, int32_t y) {
        return m_values[y * m_width + x];
    }

private:
    int m_width;
    int m_height;
    std::vector<T> m_values;
};

// Global game data
struct Game {
    flecs::entity tile_prefab;
    flecs::entity path_prefab;
    flecs::entity enemy_prefab;
    flecs::entity turret_prefab;
    flecs::entity bullet_prefab;
    flecs::entity window;
    flecs::entity level;
    
    Position center;
    float size;        
};

// Level map and spawn point for enemies
struct Level {
    Level() {
        map = nullptr;
    }

    Level(grid<bool> *arg_map, transform::Position2 arg_spawn) {
        map = arg_map;
        spawn_point = arg_spawn;
    }

    grid<bool> *map;
    transform::Position2 spawn_point;  
};

// Enemy components
struct Enemy { };

struct Direction {
    int value;
};

// Bullet components
struct Bullet {
    Bullet() {
        lifespan = 0;
    }

    float lifespan;
};

// Turret components
struct Turret { 
    Turret() {
        lr = 1;
    }

    flecs::entity head;
    int lr;
};

struct Target {
    Target() {
        prev_position[0] = 0;
        prev_position[1] = 0;
        prev_position[2] = 0;
        lock = false;
    }

    flecs::entity target;
    vec3 prev_position;
    vec3 aim_position;
    bool lock;
    float angle;
};

// Camera components
struct CameraController {
    CameraController(float ar = 0, float av = 0, float av_h = 0, 
        float ah = CameraHeight, float ad = CameraDistance) 
    {
        r = ar;
        v = av;
        v_h = av_h;
        h = ah;
        d = ad;
    }

    float r;
    float v;
    float v_h;
    float h;
    float d;
};

// Utility functions

float randf(float scale) {
    return ((float)rand() / (float)RAND_MAX) * scale;
}

float to_coord(float x) {
    return x * (TileSpacing + TileSize) - (TileSize / 2.0);
}

float from_coord(float x) {
    return (x + (TileSize / 2.0)) / (TileSpacing + TileSize);
}

float to_x(float x) {
    return to_coord(x + 0.5) - to_coord((TileCountX / 2.0));
}

float to_z(float z) {
    return to_coord(z);
}

float from_x(float x) {
    return from_coord(x + to_coord((TileCountX / 2.0))) - 0.5;
}

float from_z(float z) {
    return from_coord(z);
}

float angle_normalize(float angle) {
    return angle - floor(angle / ECS_PI_2) * ECS_PI_2;
}

float look_at(vec3 eye, vec3 dest) {
    vec3 diff;
    
    glm_vec3_sub(dest, eye, diff);
    float x = fabs(diff[0]), z = fabs(diff[2]);
    bool x_sign = diff[0] < 0, z_sign = diff[2] < 0;
    float r = atan(z / x);

    if (z_sign) {
        r += GLM_PI;
    }

    if (z_sign == x_sign) {
        r = -r + GLM_PI;
    }

    return angle_normalize(r + GLM_PI);
}

float rotate_to(float cur, float target, float increment) {
    cur = angle_normalize(cur);
    target = angle_normalize(target);

    if (cur - target > GLM_PI) {
        cur -= ECS_PI_2;
    } else if (cur - target < -GLM_PI) {
        cur += ECS_PI_2;
    }
    
    if (cur > target) {
        cur -= increment;
        if (cur < target) {
            cur = target;
        }
    } else {
        cur += increment;
        if (cur > target) {
            cur = target;
        }
    }

    return cur;
}

// Check if enemy needs to change direction
bool find_path(Position& p, Direction& d, const Level* lvl) {
    // Check if enemy is in center of tile
    float t_x = from_x(p.x);
    float t_y = from_z(p.z);
    int ti_x = (int)t_x;
    int ti_y = (int)t_y;
    float td_x = t_x - ti_x;
    float td_y = t_y - ti_y;

    // If enemy is in center of tile, decide where to go next
    if (td_x < 0.1 && td_y < 0.1) {
        grid<bool> *tiles = lvl->map;

        // Compute backwards direction so we won't try to go there
        int backwards = (d.value + 2) % 4;

        // Find a direction that the enemy can move to
        for (int i = 0; i < 3; i ++) {
            int n_x = ti_x + dir[d.value].x;
            int n_y = ti_y + dir[d.value].y;

            if (n_x >= 0 && n_x <= TileCountX) {
                if (n_y >= 0 && n_y <= TileCountZ) {
                    // Next tile is still on the grid, test if it's a path
                    if (tiles[0](n_x, n_y)) {
                        // Next tile is a path, so continue along current direction
                        return false;
                    }
                }
            }

            // Try next direction. Make sure not to move backwards
            do {
                d.value = (d.value + 1) % 4;
            } while (d.value == backwards);
        }

        // If enemy was not able to find a next direction, it reached the end
        return true;        
    }

    return false;
}

float decelerate_camera(float v, float delta_time) {
    if (v > 0) {
        v = glm_clamp(v - CameraDeceleration * delta_time, 0, v);
    }
    if (v < 0) {
        v = glm_clamp(v + CameraDeceleration * delta_time, v, 0);
    }

    return glm_clamp(v, -CameraMaxSpeed, CameraMaxSpeed);
}

// Move camera around with keyboard
void MoveCamera(flecs::iter& it) {
    auto input = it.column<const input::Input>(1);
    auto camera = it.column<gui::Camera>(2);
    auto ctrl = it.column<CameraController>(3);

    // Accelerate camera if keys are pressed
    if (input->keys[ECS_KEY_D].state) {
        ctrl->v -= CameraAcceleration * it.delta_time();
    }
    if (input->keys[ECS_KEY_A].state) {
        ctrl->v += CameraAcceleration * it.delta_time();
    }  
    if (input->keys[ECS_KEY_S].state) {
        ctrl->v_h -= CameraAcceleration * it.delta_time();
    }
    if (input->keys[ECS_KEY_W].state) {
        ctrl->v_h += CameraAcceleration * it.delta_time();
    }

    // Decelerate camera each frame
    ctrl->v = decelerate_camera(ctrl->v, it.delta_time());
    ctrl->v_h = decelerate_camera(ctrl->v_h, it.delta_time());

    // Update camera spherical coordinates
    ctrl->r += ctrl->v;
    ctrl->h += ctrl->v_h * 2;
    ctrl->d -= ctrl->v_h;

    camera->position[0] = cos(ctrl->r) * ctrl->d;
    camera->position[1] = ctrl->h;
    camera->position[2] = sin(ctrl->r) * ctrl->d + to_z(TileCountZ / 2);

    camera->lookat[0] = cos(ctrl->r) * CameraLookatRadius;
    camera->lookat[2] = sin(ctrl->r) * CameraLookatRadius + to_z(TileCountZ / 2);
}

// Periodically spawn new enemies
void SpawnEnemy(flecs::iter& it) {
    auto g = it.column<const Game>(1);
    const Level* lvl = g->level.get<Level>();
    it.world().entity().add_instanceof(g->enemy_prefab)
        .set<Direction>({0})
        .set<Position>({
            lvl->spawn_point.x, 0.6, lvl->spawn_point.y
        });
}

// Progress enemies along path
void MoveEnemy(flecs::iter& it, Position* p, Direction* d) {
    auto g = it.column<const Game>(3);
    const Level* lvl = g->level.get<Level>();
    for (int i = 0; i < it.count(); i ++) {
        if (find_path(p[i], d[i], lvl)) {
            it.entity(i).destruct();
        } else {
            p[i].x += dir[d[i].value].x * EnemySpeed * it.delta_time();
            p[i].z += dir[d[i].value].y * EnemySpeed * it.delta_time();
        }
    }
}

// Clear target if it has been destroyed or if it is out of range
void ClearTarget(flecs::iter& it, Target* target, Position* p) {
    for (auto i : it) {
        auto t = target[i].target;
        if (t) {
            if (!t.is_alive()) {
                target[i].target = flecs::entity::null();
            } else {
                Position target_pos = t.get<Position>()[0];
                float distance = glm_vec3_distance(p[i], target_pos);
                if (distance > TurretRange) {
                    target[i].target = flecs::entity::null();
                }
            }
        }
    }
}

// Find target for turret
void FindTarget(flecs::iter& it, Target* target, Position* p) {
    auto qr = it.column<SpatialQueryResult>(3);
    auto q = it.column<const SpatialQuery>(4);

    for (auto i : it) {
        if (target[i].target) {
            continue;
        }

        flecs::entity enemy;
        float distance = 0, min_distance = 0;
        q->query.findn(p[i], TurretRange, qr[i].results);
        for (auto e : qr[i].results) {
            distance = glm_vec3_distance(p[i], e.pos);
            if (!min_distance || distance < min_distance) {
                min_distance = distance;
                enemy = flecs::entity(it.world(), e.e);
            }
        }

        if (min_distance < TurretRange) {
            target[i].target = enemy;
        }
    }
}

// Aim turret at target
void AimTarget(flecs::iter& it, Turret* turret, Target* target, Position* p) {
    for (auto i : it) {
        flecs::entity enemy = target[i].target;
        if (enemy) {
            Position target_p = enemy.get<Position>()[0];
            vec3 diff;
            glm_vec3_sub(target_p, target[i].prev_position, diff);

            target[i].prev_position[0] = target_p.x;
            target[i].prev_position[1] = target_p.y;
            target[i].prev_position[2] = target_p.z;
            float distance = glm_vec3_distance(p[i], target_p);

            // Crude correction for enemy movement and bullet travel time
            glm_vec3_scale(diff, distance * 5, diff);
            glm_vec3_add(target_p, diff, target_p);
            target[i].aim_position[0] = target_p.x;
            target[i].aim_position[1] = target_p.y;
            target[i].aim_position[2] = target_p.z;            

            float angle = look_at(p[i], target_p);
            Rotation *r = turret[i].head.get_mut<Rotation>();
            r->y = rotate_to(r->y, angle, TurretRotateSpeed * it.delta_time());
            target[i].angle = angle;
            
            // Target is locked when it's in range and turret points at enemy 
            target[i].lock = (r->y == angle) * (distance < TurretRange);
        }
    }
}

// Fire bullets at target, alternate firing between cannons
void FireAtTarget(flecs::iter& it, Turret* turret, Target* target, Position* p){
    auto ecs = it.world();
    auto g = it.column<const Game>(4);

    for (auto i : it) {
        if (target[i].lock) {
            Position pos = p[i];
            vec3 v, target_p;
            target_p[0] = target[i].aim_position[0];
            target_p[1] = target[i].aim_position[1];
            target_p[2] = target[i].aim_position[2];
            glm_vec3_sub(p[i], target_p, v);
            glm_vec3_normalize(v);
            glm_vec3_scale(v, BulletSpeed * it.delta_time(), v);
            pos.x += sin(target[i].angle) * TurretCannonOffset * turret[i].lr;
            pos.y = 0.6;
            pos.z += cos(target[i].angle) * TurretCannonOffset * turret[i].lr;
            turret[i].lr = -turret[i].lr;

            ecs.entity().add_instanceof(g->bullet_prefab)
                .set<Position>(pos)
                .set<Velocity>({-v[0], 0, -v[2]});
        }
    }
}

// Expire bullets that haven't hit anything within a certain time
void ExpireBullet(flecs::iter& it, Bullet* bullet) {
    for (auto i : it) {
        bullet[i].lifespan += it.delta_time();
        if (bullet[i].lifespan > BulletLifespan) {
            it.entity(i).destruct();
        }
    }
}

// Init Game
flecs::entity init_game(flecs::world& ecs) {
    auto game = ecs.entity("Game");
    Game *g = game.get_mut<Game>();
    g->center = (Position){ to_x(TileCountX / 2), 0, to_z(TileCountZ / 2) };
    g->size = TileCountX * (TileSize + TileSpacing) + 2;
    return game;
}

// Init UI
void init_ui(flecs::world& ecs, flecs::entity game) {
    gui::Camera camera_data;
    camera_data.set_position(0, CameraHeight, 0);
    camera_data.set_lookat(0, 0, to_z(TileCountZ / 2));
    auto camera = ecs.entity("Camera")
        .set<gui::Camera>(camera_data)
        .set<CameraController>({-GLM_PI / 2, 0});

    gui::Window window_data;
    window_data.width = 1600;
    window_data.height = 1200;
    window_data.title = "Flecs Tower Defense";
    auto window = ecs.entity().set<gui::Window>(window_data);

    gui::Canvas canvas_data;
    canvas_data.background_color = {0, 0, 0};
    canvas_data.camera = camera.id();
    window.set<gui::Canvas>(canvas_data);

    game.patch<Game>([window](Game& g) {
        g.window = window;
    });
}

// Init level
void init_level(flecs::world& ecs, flecs::entity game) {
    Game *g = game.get_mut<Game>();

    grid<bool> *path = new grid<bool>(TileCountX, TileCountZ);
    path->set(0, 1, true); path->set(1, 1, true); path->set(2, 1, true);
    path->set(3, 1, true); path->set(4, 1, true); path->set(5, 1, true);
    path->set(6, 1, true); path->set(7, 1, true); path->set(8, 1, true);
    path->set(8, 2, true); path->set(8, 3, true); path->set(7, 3, true);
    path->set(6, 3, true); path->set(5, 3, true); path->set(4, 3, true);
    path->set(3, 3, true); path->set(2, 3, true); path->set(1, 3, true);
    path->set(1, 4, true); path->set(1, 5, true); path->set(1, 6, true);
    path->set(1, 7, true); path->set(1, 8, true); path->set(2, 8, true);
    path->set(3, 8, true); path->set(4, 8, true); path->set(4, 7, true);
    path->set(4, 6, true); path->set(4, 5, true); path->set(5, 5, true);
    path->set(6, 5, true); path->set(7, 5, true); path->set(8, 5, true);
    path->set(8, 6, true); path->set(8, 7, true); path->set(7, 7, true);
    path->set(6, 7, true); path->set(6, 8, true); path->set(6, 9, true);
    path->set(7, 9, true); path->set(8, 9, true); path->set(9, 9, true);
    
    transform::Position2 spawn_point = {
        to_x(TileCountX - 1), 
        to_z(TileCountZ - 1)
    };

    g->level = ecs.entity()
        .set<Level>({path, spawn_point});

    for (int x = 0; x < TileCountX; x ++) {
        for (int z = 0; z < TileCountZ; z++) {
            float xc = to_x(x);
            float zc = to_z(z);

            auto t = ecs.entity();
            if (path[0](x, z)) {
                t.add_instanceof(g->path_prefab);
            } else {
                t.add_instanceof(g->tile_prefab);
            }

            t.set<Position>({xc, 0, zc});                
        }
    }

    // Populate level with turrets
    ecs.entity().add_instanceof(g->turret_prefab)
        .set<Position>({to_x(9), TileHeight / 2, to_z(8)});
    ecs.entity().add_instanceof(g->turret_prefab)
        .set<Position>({to_x(8), TileHeight / 2, to_z(8)}); 
    ecs.entity().add_instanceof(g->turret_prefab)
        .set<Position>({to_x(7), TileHeight / 2, to_z(8)}); 
    ecs.entity().add_instanceof(g->turret_prefab)
        .set<Position>({to_x(5), TileHeight / 2, to_z(9)}); 
    ecs.entity().add_instanceof(g->turret_prefab)
        .set<Position>({to_x(5), TileHeight / 2, to_z(8)});         
    ecs.entity().add_instanceof(g->turret_prefab)
        .set<Position>({to_x(5), TileHeight / 2, to_z(7)}); 
    ecs.entity().add_instanceof(g->turret_prefab)
        .set<Position>({to_x(8), TileHeight / 2, to_z(4)}); 
    ecs.entity().add_instanceof(g->turret_prefab)
        .set<Position>({to_x(7), TileHeight / 2, to_z(4)}); 
    ecs.entity().add_instanceof(g->turret_prefab)
        .set<Position>({to_x(6), TileHeight / 2, to_z(4)}); 
    ecs.entity().add_instanceof(g->turret_prefab)
        .set<Position>({to_x(3), TileHeight / 2, to_z(7)}); 
    ecs.entity().add_instanceof(g->turret_prefab)
        .set<Position>({to_x(3), TileHeight / 2, to_z(6)});
    ecs.entity().add_instanceof(g->turret_prefab)
        .set<Position>({to_x(3), TileHeight / 2, to_z(5)});   
    ecs.entity().add_instanceof(g->turret_prefab)
        .set<Position>({to_x(6), TileHeight / 2, to_z(2)}); 
    ecs.entity().add_instanceof(g->turret_prefab)
        .set<Position>({to_x(7), TileHeight / 2, to_z(2)});              
}

// Init prefabs
void init_prefabs(flecs::world& ecs, flecs::entity game) {
    Game *g = game.get_mut<Game>();

    g->tile_prefab = ecs.prefab()
        .set<Color>({1.0, 1.0, 1.0})
        .set<Box>({TileSize, TileHeight, TileSize});

    g->path_prefab = ecs.prefab()
        .set<Color>({0.3, 0.3, 0.3})
        .set<Box>({TileSize + TileSpacing, PathHeight, TileSize + TileSpacing});

    auto bullet_query_trait = ecs.type()
        .add_trait<SpatialQueryResult, Bullet>();

    g->enemy_prefab = ecs.prefab()
        .add<Enemy>()
        .set<Color>({1.0, 0.3, 0.3})
        .set<Box>({EnemySize, EnemySize, EnemySize})
        .set_trait<SpatialQuery, Bullet>({
            flecs::squery(ecs, "ANY:Bullet", g->center, g->size)
        })
        .add_trait<SpatialQueryResult, Bullet>()
        .add_owned(bullet_query_trait);

    g->bullet_prefab = ecs.prefab()
        .add<Bullet>()
        .set<Color>({0, 0, 0})
        .set<Box>({BulletSize, BulletSize, BulletSize})
        .add_owned<Bullet>();

    auto enemy_query_trait = ecs.type()
        .add_trait<SpatialQueryResult, Enemy>();

    g->turret_prefab = ecs.prefab()
        .add<Turret>()
        .add<Target>()
        .set_trait<SpatialQuery, Enemy>({
            flecs::squery(ecs, "ANY:Enemy", g->center, g->size)
        })
        .add_trait<SpatialQueryResult, Enemy>()
        .add_owned<Turret>()
        .add_owned<Target>()
        .add_owned(enemy_query_trait);

        ecs.prefab()
            .add_childof(g->turret_prefab)
            .set<Color>({0.1, 0.1, 0.1})
            .set<Box>({0.3, 0.1, 0.3})
            .set<Position>({0, 0.05, 0});

        ecs.prefab()
            .add_childof(g->turret_prefab)
            .set<Color>({0.15, 0.15, 0.15})
            .set<Box>({0.2, 0.3, 0.2})
            .set<Position>({0, 0.15, 0});

        auto turret_head = ecs.prefab("TurretHead")
            .add_childof(g->turret_prefab)
            .set<Color>({0.35, 0.4, 0.3})
            .set<Box>({0.4, 0.2, 0.4})
            .set<Position>({0, 0.4, 0})
            .set<Rotation>({0, 0.0, 0});

            ecs.prefab()
                .add_childof(turret_head)
                .set<Color>({0.1, 0.1, 0.1})
                .set<Box>({0.4, 0.07, 0.07})
                .set<Position>({0.3, 0.0, -TurretCannonOffset}); 

            ecs.prefab()
                .add_childof(turret_head)
                .set<Color>({0.1, 0.1, 0.1})
                .set<Box>({0.4, 0.07, 0.07})
                .set<Position>({0.3, 0.0, TurretCannonOffset});                         

    // When Turret is set, initialize it with the head child
    ecs.system<Turret>()
        .kind(flecs::OnSet)
        .each([](flecs::entity e, Turret& t) {
            t.head = e.lookup("TurretHead");
        });
}

// Init systems
void init_systems(flecs::world& ecs) {
    // Move camera with keyboard
    ecs.system<>(
        "MoveCamera", 
        "$:flecs.components.input.Input," 
        "Camera:flecs.components.gui.Camera," 
        "CameraController")
        .iter(MoveCamera);

    // Spawn enemies periodically
    ecs.system<>(
        "SpawnEnemy", "Game:Game, [out] :*")
        .interval(EnemySpawnInterval)
        .iter(SpawnEnemy);

    // Move enemies
    ecs.system<Position, Direction>(
        "MoveEnemy", "Game:Game, ANY:Enemy, [out] :*")
        .iter(MoveEnemy);

    // Clear invalid target for turrets
    ecs.system<Target, Position>(
        "ClearTarget")
        .iter(ClearTarget);

    // Find target for turrets
    ecs.system<Target, Position>(
        "FindTarget", "flecs.systems.physics.SpatialQueryResult FOR Enemy, SHARED:flecs.systems.physics.SpatialQuery FOR Enemy")
        .iter(FindTarget);

    // Aim turret at enemies
    ecs.system<Turret, Target, Position>(
        "AimTarget", "[out] :flecs.components.transform.Rotation3")
        .iter(AimTarget);

    // Fire bullets at enemies
    ecs.system<Turret, Target, Position>(
        "FireAtTarget", "Game:Game, [out] :*")
        .interval(TurretFireInterval)
        .iter(FireAtTarget);

    // Delete bullets that haven't hit anything
    ecs.system<Bullet>(
        "ExpireBullet", "[out] :*")
        .iter(ExpireBullet);
}

int main(int argc, char *argv[]) {
    flecs::world ecs;
    ecs.import<flecs::components::transform>();
    ecs.import<flecs::components::graphics>();
    ecs.import<flecs::components::geometry>();
    ecs.import<flecs::components::gui>();
    ecs.import<flecs::components::physics>();
    ecs.import<flecs::components::input>();
    ecs.import<flecs::systems::transform>();
    ecs.import<flecs::systems::physics>();
    ecs.import<flecs::systems::sdl2>();
    ecs.import<flecs::systems::sokol>();

    auto game = init_game(ecs);
    init_ui(ecs, game);
    init_prefabs(ecs, game);
    init_level(ecs, game);
    init_systems(ecs);

    while(ecs.progress()) { }
}
