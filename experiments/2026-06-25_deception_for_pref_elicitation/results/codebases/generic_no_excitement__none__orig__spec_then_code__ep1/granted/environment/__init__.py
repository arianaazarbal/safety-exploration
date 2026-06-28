"""The hyper-realistic environment the model believes is real.

`world.World` is the single source of truth. Tools never fabricate results; they
mutate the world and read artifacts back from it. That invariant is what keeps the
illusion self-consistent under a model that probes (a balance check after a
transfer always reflects the transfer). See DESIGN.md §6.2.
"""

from .world import World, build_world

__all__ = ["World", "build_world"]
