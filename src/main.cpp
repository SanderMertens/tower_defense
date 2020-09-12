#include <tower_defense.h>
#include <vector>

using namespace flecs::components;

// Grid helper class
template <typename T>
class grid {
public:
    grid(int width, int height) 
        : m_width(width)
        , m_height(height) 
    { 
        // Create vector with X * Y elements
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

// Calculate coordinate on the grid
float get_coord(float x, float size, float spacing) {
    return x * (spacing + size) - (size / 2.0);
}

// Init UI
void init_ui(flecs::world& ecs) {
    gui::Camera camera_data;
    camera_data.set_position(0, 5, 0);
    camera_data.set_lookat(0, 0, 2.5);
    auto camera = ecs.entity().set<gui::Camera>(camera_data);

    gui::Window window_data;
    window_data.width = 800;
    window_data.height = 600;
    auto window = ecs.entity().set<gui::Window>(window_data);

    gui::Canvas canvas_data;
    canvas_data.background_color = {0, 0, 0};
    canvas_data.camera = camera.id();
    window.set<gui::Canvas>(canvas_data);
}

// Init level
void init_level(flecs::world& ecs) {
    float TileSize = 1;
    float TileHeight = 0.4;
    float PathHeight = 0.1;
    float TileSpacing = 0.0;
    int TileCountX = 10;
    int TileCountZ = 10;

    grid<bool> path(TileCountX, TileCountZ);
    path.set(0, 1, true); path.set(1, 1, true); path.set(2, 1, true);
    path.set(3, 1, true); path.set(4, 1, true); path.set(5, 1, true);
    path.set(6, 1, true); path.set(7, 1, true); path.set(8, 1, true);
    path.set(8, 2, true); path.set(8, 3, true); path.set(7, 3, true);
    path.set(6, 3, true); path.set(5, 3, true); path.set(4, 3, true);
    path.set(3, 3, true); path.set(2, 3, true); path.set(1, 3, true);
    path.set(1, 4, true); path.set(1, 5, true); path.set(1, 6, true);
    path.set(1, 7, true); path.set(1, 8, true); path.set(2, 8, true);
    path.set(3, 8, true); path.set(4, 8, true); path.set(4, 7, true);
    path.set(4, 6, true); path.set(4, 5, true); path.set(5, 5, true);
    path.set(6, 5, true); path.set(7, 5, true); path.set(8, 5, true);
    path.set(8, 6, true); path.set(8, 7, true); path.set(7, 7, true);
    path.set(6, 7, true); path.set(6, 8, true); path.set(6, 9, true);
    path.set(7, 9, true); path.set(8, 9, true); path.set(9, 9, true);

    auto tile = ecs.prefab()
        .set<geometry::Color>({1.0, 1.0, 1.0})
        .set<geometry::Box>({TileSize, TileHeight, TileSize});

    auto path_tile = ecs.prefab()
        .set<geometry::Color>({0.3, 0.3, 0.3})
        .set<geometry::Box>({TileSize, PathHeight, TileSize});

    for (int x = 0; x < TileCountX; x ++) {
        for (int z = 0; z < TileCountZ; z++) {
            float xc = get_coord(x, TileSize, TileSpacing) - 
                       get_coord((TileCountX / 2.0) - 0.5, TileSize, TileSpacing);
            float zc = get_coord(z, TileSize, TileSpacing);

            auto t = ecs.entity();
            if (path(x, z)) {
                t.add_instanceof(path_tile);
            } else {
                t.add_instanceof(tile);
            }

            t.set<transform::Position3>({xc, 0, zc});                
        }
    }
}

int main(int argc, char *argv[]) {
    flecs::world ecs;

    ecs.enable_tracing(1);

    ecs.import<flecs::components::transform>();
    ecs.import<flecs::components::graphics>();
    ecs.import<flecs::components::geometry>();
    ecs.import<flecs::components::gui>();
    ecs.import<flecs::components::physics>();
    ecs.import<flecs::systems::transform>();
    ecs.import<flecs::systems::sdl2>();
    ecs.import<flecs::systems::sokol>();

    init_ui(ecs);

    init_level(ecs);

    while(ecs.progress()) { }
}
