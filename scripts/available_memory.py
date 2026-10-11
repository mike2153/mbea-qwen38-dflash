"""Budget vLLM separately from runtime overhead and a non-negotiable Windows reserve."""
import math
import os
import sys

GIB = 1024**3


def memory_fraction(free, total, reserve=1.0 * GIB, overhead=2.2 * GIB,
                    floor=27.35 * GIB, cap=0.95):
    values = (free, total, reserve, overhead, floor, cap)
    if (not all(math.isfinite(x) for x in values) or not 0 < free <= total
            or reserve < .5 * GIB or overhead < 2.2 * GIB or floor < 0 or not 0 < cap < 1):
        raise ValueError('Invalid memory policy: keep at least 0.5 GiB Windows reserve')
    fraction = math.floor(min(free - overhead - reserve, total * cap) / total * 10000) / 10000
    if fraction <= 0 or fraction * total < floor:
        raise ValueError('Insufficient VRAM for the context floor and Windows reserve; close GPU apps')
    return fraction


def main():
    import torch
    reserve = float(os.environ.get('VRAM_RESERVE_GIB', '1.0')) * GIB
    cap = float(os.environ.get('VRAM_CAP', '0.95'))
    torch.cuda.init()
    free, total = torch.cuda.mem_get_info(0)
    fraction = memory_fraction(free, total, reserve=reserve, cap=cap)
    args = sys.argv[1:]
    args[args.index('--gpu-memory-utilization') + 1] = str(fraction)
    print(f'[memory-budget] free={free/GIB:.2f} GiB total={total/GIB:.2f} GiB '
          f'budget={fraction*total/GIB:.2f} GiB overhead=2.20 GiB '
          f'expected_windows_free={(free-fraction*total-2.2*GIB)/GIB:.2f} GiB '
          f'utilization={fraction}', flush=True)
    os.execv('/opt/radiance_entrypoint.sh', ['/opt/radiance_entrypoint.sh', *args])


if __name__ == '__main__':
    main()
