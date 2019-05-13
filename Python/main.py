from onelevel15to1 import cost_of_one_level_15to1
from twolevel15to1 import cost_of_two_level_15to1
from twolevel20to4 import cost_of_two_level_20to4
from twolevel8toCCZ import cost_of_two_level_8toccz
from smallfootprint import cost_of_one_level_15to1_small_footprint
from smallfootprint import cost_of_two_level_15to1_small_footprint

# You can use the following functions to calculate resource costs:
# cost_of_one_level_15to1(pphys, dx, dz, dm)
# cost_of_two_level_15to1(pphys, dx, dz, dm, dx2, dz2, dm2, nl1)
# cost_of_two_level_20to4(pphys, dx, dz, dm, dx2, dz2, dm2, nl1)
# cost_of_two_level_8toccz(pphys, dx, dz, dm, dx2, dz2, dm2, nl1)
# cost_of_one_level_15to1_small_footprint(pphys, dx, dz, dm)
# cost_of_two_level_15to1_small_footprint(pphys, dx, dz, dm, dx2, dz2, dm2)

print('----- pphys = 10^(-4) -----')
cost_of_one_level_15to1(0.0001, 7, 3, 3)
cost_of_one_level_15to1(0.0001, 9, 3, 3)
cost_of_one_level_15to1(0.0001, 11, 5, 5)
cost_of_two_level_20to4(0.0001, 7, 3, 3, 15, 7, 9, 4)
# cost_of_two_level_15to1(0.0001,9,3,3,25,9,9,4)

print('----- pphys = 10^(-3) -----')
cost_of_one_level_15to1(0.001, 17, 7, 7)
cost_of_two_level_20to4(0.001, 13, 5, 5, 21, 13, 13, 6)
cost_of_two_level_20to4(0.001, 13, 5, 5, 25, 13, 15, 4)
cost_of_two_level_15to1(0.001, 11, 5, 5, 25, 11, 11, 6)
cost_of_two_level_15to1(0.001, 11, 5, 5, 29, 13, 13, 6)
# cost_of_two_level_15to1(0.001,15,7,7,41,17,17,6)

print('----- CCZ synthillation -----')
cost_of_two_level_8toccz(0.0001, 7, 3, 3, 15, 7, 9, 4)
cost_of_two_level_8toccz(0.001, 13, 7, 7, 23, 13, 15, 6)

print('----- Small footprint -----')
cost_of_one_level_15to1_small_footprint(0.0001, 9, 3, 3)
cost_of_two_level_15to1_small_footprint(0.001, 9, 5, 5, 21, 9, 9)

print('----- Use Mathematica notebook for increased precision! -----')
print('The numbers in this Python notebook become unreliable for output errors smaller than 10^(-13)')
print('due to the limit of machine-precision numbers. In principle, it is possible to implement')
print('arbitrary-precision arithmetic with complex numbers in Python, but it is a bit of a hassle.')
print('This work was originally done in Mathematica, which natively supports arbitrary-precision arithmetic.')
print('If you require higher precision, use the Mathematica notebook.')
