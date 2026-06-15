from enum import Enum, auto

import numpy as np


class Constants:
    mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (N/A^2)


class YoungsModulus(Enum):
    """
    Enum representing Young's modulus (E) for different materials. Info can be found here:
    https://github.com/google-deepmind/mujoco/issues/827
    https://en.wikipedia.org/wiki/Young%27s_modulus

    Attributes
    ----------
    STEEL : float
        Young's modulus of steel in Pascals (Pa).
    RUBBER : float
        Young's modulus of rubber in Pascals (Pa).
    COPPER : float
        Young's modulus of copper in Pascals (Pa).
    """

    STEEL = 200e9
    RUBBER = 0.1e9  # somewhere between 0.01–0.1
    COPPER = 110e9


class ShearModulus(Enum):
    """
    Enum representing Shear modulus (G) for different materials. Info can be found here:
    https://github.com/google-deepmind/mujoco/issues/827
    https://en.wikipedia.org/wiki/Young%27s_modulus

    Attributes
    ----------
    STEEL : float
        Shear modulus of steel in Pascals (Pa).
    RUBBER : float
        Shear modulus of rubber in Pascals (Pa).
    COPPER : float
        Shear modulus of copper in Pascals (Pa).
    """

    STEEL = 79.3e9
    RUBBER = 0.0006e9
    COPPER = 44.7e9


class Material(Enum):
    """
    Enum representing material types.

    Attributes
    ----------
    STEEL : int
        Represents the material as steel.
    RUBBER : int
        Represents the material as rubber.
    COPPER : int
        Represents the material as copper.
    """

    STEEL = 0
    RUBBER = 1
    COPPER = 2


def cable_properties(material: Material) -> dict:
    """
    Computes the twist and bend stiffness for a given material.

    Parameters
    ----------
    material : Material
        The material of the cable. Use the Material enum (e.g., STEEL, RUBBER, COPPER).

    Returns
    -------
    dict
        A dictionary containing:
        - 'twist': The shear modulus (representing twist stiffness) in Pascals (Pa).
        - 'bend': The Young's modulus (representing bend stiffness) in Pascals (Pa).

    Notes
    -----
    - The twist stiffness is directly represented by the shear modulus (G) of the material.
    - The bend stiffness is directly represented by Young's modulus (E) of the material.
    - The calculation assumes normalized stiffnesses without explicit dependence on cross-section dimensions.
    """

    # Get the correct Young's modulus and Shear modulus for the given material
    youngs_modulus = YoungsModulus[material.name].value
    shear_modulus = ShearModulus[material.name].value

    return {
        "twist": shear_modulus,  # In Pascals (Pa)
        "bend": youngs_modulus,  # In Pascals (Pa)
    }


def biot_savarts_law(
    point: np.ndarray, I_wire: float, wire_position: np.ndarray = None
) -> np.ndarray:
    """
    Compute the magnetic field at a point due to a current-carrying wire.

    This function calculates the magnetic field at a given point in space, assuming an infinitely long wire
    carrying a steady current. The magnetic field is computed using the Biot-Savart law.

    Parameters
    ----------
    point : np.ndarray
        The coordinates of the point where the magnetic field is computed. Can be 2D (x, y) or 3D (x, y, z).
    I_wire : float
        The current flowing through the wire located at the origin or a custom position.
    wire_position : np.ndarray, optional
        The coordinates of the wire (default is at the origin). Should be of the same dimension as `point`.

    Returns
    ----------
    np.ndarray
        A vector representing the magnetic field components at the given point. Returns a 2D vector for 2D inputs
        and a 3D vector for 3D inputs.

    Notes
    -----
    - If the point is located at the wire's position (r = 0), the magnetic field is undefined and returns [0, 0] or [0, 0, 0].
    - For 2D inputs, the magnetic field is directed perpendicular to the radial vector in the xy-plane.
    - For 3D inputs, the wire is assumed to be infinitely long along the z-axis, and the field is calculated in the xy-plane.
    """
    if wire_position is None:
        wire_position = np.zeros_like(point)

    r_vec = point - wire_position
    r = np.linalg.norm(r_vec)

    if r == 0:
        return np.zeros_like(point)

    # Biot-Savart law in 2D and 3D
    if len(point) == 2:
        B_magnitude = (Constants.mu_0 * I_wire) / (2 * np.pi * r)
        B_direction = (
            np.array([-r_vec[1], r_vec[0]]) / r
        )  # Perpendicular to radial vector
        return B_magnitude * B_direction
    elif len(point) == 3:
        B_magnitude = (Constants.mu_0 * I_wire) / (2 * np.pi * r)
        B_direction = np.cross([0, 0, 1], r_vec) / r  # Wire is along the z-axis
        return B_magnitude * B_direction
    else:
        raise ValueError("Input point must be either 2D or 3D.")


def amperes_law(
    loop_points: np.ndarray, I_wire: float, wire_position: np.ndarray = None
) -> float:
    """
    Compute the current enclosed by a closed loop using Ampère's law.

    This function calculates the total current enclosed by a loop of points in space, based on the circulation of
    the magnetic field along the loop. The magnetic field is generated by a current-carrying wire, and the current
    is computed using Ampère's law.

    Parameters
    ----------
    loop_points : np.ndarray
        An array of shape (N, 2) or (N, 3) containing the coordinates of points forming a closed loop.
    I_wire : float
        The current through the wire generating the magnetic field.
    wire_position : np.ndarray, optional
        The position of the wire (default is at the origin). Should be of the same dimension as `loop_points`.

    Returns
    ----------
    float
        The total enclosed current through the loop.

    Notes
    -----
    - The magnetic field at each point in the loop is calculated using the Biot-Savart law.
    - Ampère's law relates the circulation of the magnetic field around the loop to the total current enclosed by the loop.
    - This function supports both 2D and 3D loops, depending on the dimensionality of `loop_points`.
    """
    if wire_position is None:
        wire_position = np.zeros(loop_points.shape[1])

    N = len(loop_points)
    total_circulation = 0

    for i in range(N):
        point_i = loop_points[i]
        point_ip1 = loop_points[(i + 1) % N]

        # Compute the magnetic field at point (x_i, y_i, [z_i])
        B_i = biot_savarts_law(point_i, I_wire, wire_position)

        # Compute the line segment vector (dl)
        dl = point_ip1 - point_i

        # Dot product of B_i and dl
        total_circulation += np.dot(B_i, dl)

    # Use Ampère's law to calculate the enclosed current
    I_enc = total_circulation / Constants.mu_0
    return I_enc


class ContactModelType(Enum):
    HARD = auto()
    SOFT = auto()
    PwoF = auto()
