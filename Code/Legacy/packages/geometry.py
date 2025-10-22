"""Geometrical handling of spins ordering"""

import torch

def spiral_contour(s, R, L):
    """Get the contour as a counter-clockwise spiral.
    -s: the starting spin index,
    -R: is the linear size of the patch,
    -L: is the linear size of the system.
    Handles correctly periodic boundary conditions.
    Spin indexes are assumed to go row-wise from left to right.
    """
    with torch.no_grad():
        d = torch.arange(L*L).reshape(L,L)  # Array we are going to use to generate spin indexes
        seed = [s // L, s % L]  # Get [x, y] position of the seed
        up_row = (seed[0] - 1) % L
        bot_row = (seed[0] + R) % L
        left_column = (seed[1] - 1) % L
        right_column = (seed[1] + R) % L

        # Extracting the spin indexes based on the patch size and handling periodic boundary conditions

        # Left side
        if up_row < bot_row:
            consider = d[up_row:bot_row, left_column]
        else:
            consider = torch.cat((d[up_row:, left_column], d[:bot_row, left_column]))

        # Bottom side
        if left_column < right_column:
            consider = torch.cat((consider, d[bot_row, left_column:right_column]))
        else:
            consider = torch.cat((consider, d[bot_row, left_column:], d[bot_row, :right_column]))

        # Right side
        if up_row < bot_row:
            consider = torch.cat((consider, torch.flip(d[up_row + 1:bot_row + 1, right_column], (0,))))
        else:
            consider = torch.cat((consider, torch.flip(d[:bot_row + 1, right_column], (0,)),
                                 torch.flip(d[up_row + 1:, right_column], (0,))))

        # Top side
        if left_column < right_column:
            consider = torch.cat((consider, torch.flip(d[up_row, left_column + 1:right_column + 1], (0,))))
        else:
            consider = torch.cat((consider, torch.flip(d[up_row, :right_column + 1], (0,)),
                                 torch.flip(d[up_row, left_column + 1:], (0,))))

        return consider.int()

def stripe_contour(s, R, L):
    """Get the contour as a counter-clockwise spiral.
    -s: the starting spin index,
    -R: width of the stripe
    -L: is the linear size of the system.
    Handles correctly periodic boundary conditions.
    Spin indexes are assumed to go row-wise from left to right.
    """
    with torch.no_grad():
        d = torch.arange(L*L).reshape(L,L)  # Array we are going to use to generate spin indexes
        seed = [s // L, s % L]  # Get [x, y] position of the seed
        left_column = (seed[1] - 1) % L
        right_column = (seed[1] + R) % L
        consider = torch.cat((d[:,left_column], d[:, right_column]))
        return consider.int()

def get_patch(s, R, L, contour_f=spiral_contour,interior_f = spiral_contour, reverse = False):
    """
    Get the ordering of a whole patch by iteratively calling a contour function.

    Parameters:
    - s: Starting spin index
    - R: Linear size of the patch
    - L: Linear size of the system
    - contour_f: Contour function to generate the ordering (default is spiral_contour)
    - interior_f: Contour function to generate the ordering (default is spiral_contour)
    - reverse: Flag to determine whether to reverse the interior ordering (default is False)

    Returns:
    - exterior: Ordering for the exterior part of the patch
    - interior: Ordering for the interior part of the patch
    """

    # Get exterior ordering using the specified contour function
    exterior = contour_f(s, R, L)

    # Initialize variables for interior ordering
    newR = int(R - 2)

    interior = torch.empty(0).int()
    # If the patch size is odd, include the center spin at the end
    if newR == -1:
        if interior_f == stripe_contour:
            news_col = s % L
            d = torch.arange(L*L).reshape(L,L)
            interior = torch.cat((interior, d[:,news_col]))
        else:      
            interior = torch.cat((interior, torch.tensor(s, dtype=int).reshape(1)))

    # Determine the next starting spin index for the interior
    if (s + 1) % L == 0:
        news = (s + 1) % (L * L)
    else:
        news = (s + L + 1) % (L * L)

    # Iterate to generate interior ordering for decreasing patch sizes
    while newR >= 0:
        interior = torch.cat((interior, interior_f(news, newR, L)))
        # Detemine new patch size
        newR -= 2
        # If the patch size is odd, include the center spin at the end
        if newR == -1:
            if interior_f == stripe_contour:
                news_col = news % L
                d = torch.arange(L*L).reshape(L,L)
                interior = torch.cat((interior, d[:,news_col]))
            else:
                interior = torch.cat((interior, torch.tensor(news, dtype=int).reshape(1)))
        # Determine the next starting spin index for the interior
        if (news + 1) % L == 0:
            news = (news + 1) % (L * L)
        else:
            news = int((news + L + 1) % (L * L))
    # Reverse the interior ordering if specified
    if not reverse:
        return exterior.long(), interior.long()
    else:
        return exterior.long(), torch.flip(interior, (0,)).long()


def angle_contour(s, R, L):
  """Get the contour starting from the angles.
  -s: the starting spin index,
  -R: is the linear size of the patch,
  -L: is the linear size of the system.
  Handles correctly periodic boundary conditions.
  Spin indexes are assumed to go row-wise from left to right."""
  # Create a 2D grid with indices from 0 to L*L-1
  d = torch.arange(L*L).reshape(L, L)

  # Define the seed point (starting point) using s, which is converted to (row, column) coordinates
  seed = [s // L, s % L]

  # Calculate the row and column indices for the surrounding region based on the seed point and radius R
  up_row = (seed[0] - 1) % L
  bot_row = (seed[0] + R) % L
  left_column = (seed[1] - 1) % L
  right_column = (seed[1] + R) % L

  # Initialize a list to store the coordinates of points to consider in the contour
  consider = []

  # Initialize loop control variables
  go = True
  count = 1

  # Define the coordinates of the top-left, top-right, bottom-right, and bottom-left corners of the region
  tl = [up_row, left_column]
  tr = [up_row, right_column]
  br = [bot_row, right_column]
  bl = [bot_row, left_column]

  # Add these corner points to the list of points to consider
  consider = [tl, tr, br, bl]

  # Continue the loop until the condition is no longer met
  while go:
    # Check and add points by propagating from the top-left angle
    add = [up_row, (left_column + count) % L]
    if add not in consider:
        consider.append(add)
    else:
        go = False

    add = [(up_row + count) % L, left_column]
    if add not in consider:
        consider.append(add)
    else:
        go = False

    # Check and add points by propagating from the top-right angle
    add = [ up_row, (right_column - count) % L]
    if add not in consider:
        consider.append(add)
    else:
        go = False

    add = [(up_row + count) % L, right_column]
    if add not in consider:
        consider.append(add)
    else:
        go = False

    # Check and add points by propagating from the bottom-right angle
    add = [bot_row, (right_column - count) % L]
    if add not in consider:
        consider.append(add)
    else:
        go = False

    add = [(bot_row - count) % L, right_column]
    if add not in consider:
        consider.append(add)
    else:
        go = False

    # Check and add points by propagating from the bottom-left angle
    add = [bot_row, (left_column + count) % L]
    if add not in consider:
        consider.append(add)
    else:
        go = False

    add = [(bot_row - count) % L, left_column]
    if add not in consider:
        consider.append(add)
    else:
        go = False

    # Increment the count for the next iteration
    count += 1

  # Return a tensor containing the values at the selected points in the 2D grid
  return torch.tensor([d[x[0]][x[1]] for x in consider])

def angleless_angle_contour(s, R, L):
  """Get the contour starting from the angles, but does not include 4 angle spins.
  -s: the starting spin index,
  -R: is the linear size of the patch,
  -L: is the linear size of the system.
  Handles correctly periodic boundary conditions.
  Spin indexes are assumed to go row-wise from left to right."""
  # Create a 2D grid with indices from 0 to L*L-1
  d = torch.arange(L*L).reshape(L, L)

  # Define the seed point (starting point) using s, which is converted to (row, column) coordinates
  seed = [s // L, s % L]

  # Calculate the row and column indices for the surrounding region based on the seed point and radius R
  up_row = (seed[0] - 1) % L
  bot_row = (seed[0] + R) % L
  left_column = (seed[1] - 1) % L
  right_column = (seed[1] + R) % L

  # Initialize a list to store the coordinates of points to consider in the contour
  consider = []

  # Initialize loop control variables
  go = True
  count = 1

  # Add these corner points to the list of points to consider
  consider = []

  # Continue the loop until the condition is no longer met
  while go:
    # Check and add points by propagating from the top-left angle
    add = [up_row, (left_column + count) % L]
    if add not in consider:
        consider.append(add)
    else:
        go = False

    add = [(up_row + count) % L, left_column]
    if add not in consider:
        consider.append(add)
    else:
        go = False

    # Check and add points by propagating from the top-right angle
    add = [ up_row, (right_column - count) % L]
    if add not in consider:
        consider.append(add)
    else:
        go = False

    add = [(up_row + count) % L, right_column]
    if add not in consider:
        consider.append(add)
    else:
        go = False

    # Check and add points by propagating from the bottom-right angle
    add = [bot_row, (right_column - count) % L]
    if add not in consider:
        consider.append(add)
    else:
        go = False

    add = [(bot_row - count) % L, right_column]
    if add not in consider:
        consider.append(add)
    else:
        go = False

    # Check and add points by propagating from the bottom-left angle
    add = [bot_row, (left_column + count) % L]
    if add not in consider:
        consider.append(add)
    else:
        go = False

    add = [(bot_row - count) % L, left_column]
    if add not in consider:
        consider.append(add)
    else:
        go = False

    # Increment the count for the next iteration
    count += 1

  # Return a tensor containing the values at the selected points in the 2D grid
  return torch.tensor([d[x[0]][x[1]] for x in consider])

def split_grid(R, L, bord_fun, int_fun,reverse = False, shifted = False):
  """Split the total grid into patches.

  Inputs:
  -L: side of the big grid
  -R side of the patches
  -fun: function to use for the patches formation (e.g. spiral_contour)
  -reverse: whether spins are given in reverse order
  -shifted: whether grid is shifted

  Returns:
  -sorted_grid: list of tuples, having the border and the paches. Order is classical row-wise order
  """

  sorted_grid = []
  #Check if values make sense
  if L % R != 0:
    raise ValueError("L must be a multiple or R in order to have an integer number of patches!")
  if not shifted:
    for i in range(round(L/R)): #just call the fun multiple times
      for j in range(round(L/R)):
        sorted_grid.append(get_patch(R*L*i+j*R,R,L,contour_f = bord_fun, interior_f=int_fun,reverse = reverse))
  else:
    for i in range(round(L/R)): #just call the fun multiple times
      for j in range(round(L/R)):
        sorted_grid.append(get_patch(R*L*i+j*R+int(R/2+R*L/2),R,L,contour_f = bord_fun, interior_f=int_fun,reverse = reverse))
  return sorted_grid