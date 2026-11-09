// Indirect-count multi-draws masked on the GPU (nvn_mtl.m mdx_run)
#include <metal_stdlib>
using namespace metal;
struct mjob { device const uint *rec; device const uint *cnt; device uint *out; uint i, words; };   // i: the draw's index in its multi-draw; for an indirect mesh draw (words 0) the mesh grid's limit
#define MDX_MESH_PARTS 8u
#define MDX_MESH_GRID 1024u
kernel void mdx_mask(device const mjob *j [[buffer(0)]], uint t [[thread_position_in_grid]]) {
  mjob m = j[t]; if (!m.out) return;
  // An indirect mesh draw: NVN's {count, first} as Metal's threadgroups {x, y, z}: {count, 1, 1} at +0 for a task shader's, and at +16 {ceil(count / limit), 1, 1} for the object
  // An indirect mesh draw: NVN's {count, first} as Metal's threadgroups {x, y, z}: {min(count, 1024), 1, 1}
  // Metal limits a mesh grid without task shader to 1024 threadgroups (DMSH_MESH_GRID). Clamped to 1024 to prevent drops and z-fighting.
  if (!m.words) {
    uint c = m.rec[0];
    uint clamped = min(c, 1024u);
    m.out[0] = clamped; m.out[1] = 1u; m.out[2] = 1u;
    m.out[4] = m.i ? c / m.i + (c % m.i != 0u) : 0u; m.out[5] = 1u; m.out[6] = 1u;
    return;
  }
  bool live = m.i < *m.cnt;
  for (uint k = 0; k < m.words; k++) m.out[k] = live ? m.rec[k] : 0u; }
