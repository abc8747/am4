# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

base_path = Path("../../../src/am4/utils/data/")
airports = pq.read_table(base_path / "airports.parquet")
aircrafts = pq.read_table(base_path / "aircrafts.parquet")
# %%
import pyarrow.csv as csv

csv.write_csv(
    airports,
    "airports.csv",
    csv.WriteOptions(
        delimiter="\t",
    ),
)
# %%
routes = pq.read_table(base_path / "routes.parquet")

yds = np.array(routes["yd"], dtype=np.uint16)
jds = np.array(routes["jd"], dtype=np.uint16)
fds = np.array(routes["fd"], dtype=np.uint16)
# %%
import numpy as np


def bit_pack_and_save(ys, js, fs, filename="test.bin"):
    n = ys.size

    total_bits = n * 33
    total_bytes = (total_bits + 7) // 8  # Round up to the nearest byte

    packed_bytes = bytearray(total_bytes)

    bit_offset = 0
    for i in range(n):
        combined = (ys[i] << 21) | (js[i] << 10) | fs[i]

        for bit in range(33):
            byte_index = (bit_offset + bit) // 8
            bit_index = (bit_offset + bit) % 8
            packed_bytes[byte_index] |= ((combined >> (32 - bit)) & 1) << (7 - bit_index)

        bit_offset += 33
        if i % 1000 == 0:
            print(f"{i}/{n}")

    with open(filename, "wb") as f:
        f.write(packed_bytes)


bit_pack_and_save(yds, jds, fds)

# %%
print(yds.min(), yds.max())
print(jds.min(), jds.max())
print(fds.min(), fds.max())
# print(sms.min())
