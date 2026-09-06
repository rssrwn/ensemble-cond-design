# **************************
# ***** Util functions *****
# **************************


# Putting these at the top since the global defs below are very long

def check_type(obj, allowed_types, name="object"):
    # Allow lists of accepted types, convert to a singleton list if single type provided
    allowed_types = [allowed_types] if not isinstance(allowed_types, list) else allowed_types
    is_type = [isinstance(obj, obj_type) for obj_type in allowed_types]

    if not any(is_type):
        type_str = " or ".join([str(t) for t in allowed_types])
        raise TypeError(f"{name} must be an instance of {type_str} or one of their subclasses, got {type(obj)}")


def check_dict_key(map, key, dict_name="dictionary"):
    if key not in map:
        raise RuntimeError(f"{dict_name} must contain key {key}")


def check_unique(obj_list, name="objects"):
    if len(obj_list) != len(set(obj_list)):
        raise RuntimeError(f"{name} cannot contain duplicates")


def check_type_all(obj_list, exp_type, name="list"):
    for obj in obj_list:
        if not isinstance(obj, exp_type):
            raise TypeError(f"all objects in {name} must be instances of {exp_type}")
        

# Note arr can be used for either np arrays or torch tensors
def check_shape_len(arr, allowed, name="object"):
    num_dims = len(arr.shape)
    allowed = [allowed] if isinstance(allowed, int) else allowed
    if num_dims not in allowed: 
        raise RuntimeError(f"Number of dimensions of {name} must be in {str(allowed)}, got {num_dims}")


# Note arr can be used for either np arrays or torch tensors
def check_dim_shape(arr, dim, allowed, name="object"):
    shape = arr.shape[dim]
    allowed = [allowed] if isinstance(allowed, int) else allowed
    if shape not in allowed:
        raise RuntimeError(f"Shape of {name} for dim {dim} must be in {allowed}, got {shape}")


# Note arr can be used for either np arrays or torch tensors
def check_shapes_equal(arr1, arr2, dims=None):
    if dims is None:
        if arr1.shape != arr2.shape:
            raise RuntimeError(f"objects must have the same shape, got {arr1.shape} and {arr2.shape}")
        else:
            return

    if isinstance(dims, int):
        dims = [dims]

    t1_dims = [arr1.shape[dim] for dim in dims]
    t2_dims = [arr2.shape[dim] for dim in dims]
    if t1_dims != t2_dims:
        raise RuntimeError(f"Expected dimensions {str(dims)} to match, got {arr1.shape} and {arr2.shape}")


# ******************************
# ***** Global definitions *****
# ******************************


PAD_TOKEN = "PAD"
MASK_TOKEN = "MASK"
LIGAND_RES_TOKEN = "LIG"

PICKLE_PROTOCOL = 4


# *** Specific Token Sets ***

# _ATOM_TYPES = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Se", "Br"]

CHARGED_ATOM_TYPES = [
    "Br_1",
    "S_1",
    "S_0",
    "F_-1",
    "O_0",
    "I_0",
    "Bi_2",
    "N_1",
    "F_0",
    "O_-1",
    "C_1",
    "C_-1",
    "Bi_0",
    "P_0",
    "H_0",
    "Cl_1",
    "Cl_0",
    "Br_0",
    "Cl_-1",
    "S_2",
    "I_1",
    "B_-1",
    "N_0",
    "N_-1",
    "Si_0",
    "P_-1",
    "B_0",
    "N_-2",
    "H_1",
    "I_2",
    "P_1",
    "O_1",
    "C_0",
    "Si_1",
    "S_-1",
    "S_3"
]

# Let -100 correspond to unknown/pad charge
ATOM_CHARGES = [-100, 0, 1, 2, 3, -1, -2, -3]

POCKET_RESIDUE_NAMES = [
    "ILE",
    "LEU",
    "ALA",
    "GLU",
    "VAL",
    "PHE",
    "CYS",
    "GLN",
    "THR",
    "SER",
    "TYR",
    "MET",
    "LYS",
    "HIS",
    "GLY",
    "PRO",
    "ARG",
    "ASP",
    "ASN",
    "CME",
    "TRP",
    "LLP",
    "OAS",
    "SGB",
    "CSD",
    "SEP",
    "OCY",
    "TIS",
    "SCY",
    "OCS",
    "QPA",
    "KPI",
    "PHD",
    "MEN",
    "SUN",
    "TPO",
    "CSO",
    "YCM",
    "UNK",
    "ALY",
    "SVX",
    "PTR",
    "KCX",
    "HOX",
    "PCA",
    "00C",
    "DM0",
    "XCN",
    "SXE",
    "HYP",
    "CSX",
    "CSS",
    "ORN",
    "MLY",
    "2CO",
    "NEP",
    "YOF",
    "SNN",
]
