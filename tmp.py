# import time

# the_time = time.time()

# time.sleep(5)

# elapsed_time = time.time() - the_time
# hours, rem = divmod(elapsed_time, 3600)
# minutes, seconds = divmod(rem, 60)
# print(f"{int(hours):02}:{int(minutes):02}:{seconds:05.2f}")

# print(11 * 60 / 6)

# import numpy as np

# print(np.log(0))

# import pandas as pd

# df = pd.DataFrame({'A': [1, 2, 3], 'B': [[1, 2, 3], 'foo', []]})
# print(df)
# df_exploded = df.explode('B')
# print(df_exploded)

# print(2000000)
# print(type(2000000))

# print(2_000_000)
# print(type(2_000_000))

# print(128 + 64)
value = 0.2

label, opp_label = ("OPEN", "CLOSE") if value > 0.5 else ("CLOSE", "OPEN")

print(label, opp_label)