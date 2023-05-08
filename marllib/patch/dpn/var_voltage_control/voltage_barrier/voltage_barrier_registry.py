from .bowl import bowl
from .bump import bump
from .courant_beltrami import courant_beltrami
from .l1 import l1
from .l2 import l2



Voltage_Barrier = dict(
    l1=l1,
    l2=l2,
    bowl=bowl,
    bump=bump,
    courant_beltrami=courant_beltrami
)