import os

def path_to_parents(levels=1):
    """
    Change the current working directory to its parent directory.
    
    This function navigates up the directory tree by the specified number of levels,
    changing the current working directory to the parent (or ancestor) directory.
    This is equivalent to executing 'cd ../' or 'cd ../../' commands depending on the levels.
    
    Args:
        levels (int, optional): Number of levels to go up in the directory tree. 
            Defaults to 1. For example:
            - levels=1: Go up one level (equivalent to 'cd ../')
            - levels=2: Go up two levels (equivalent to 'cd ../../')
            - levels=3: Go up three levels (equivalent to 'cd ../../../')
    
    Returns:
        None: This function modifies the current working directory in-place.
    
    Side Effects:
        - Changes the current working directory
        - Prints the new working directory path to stdout
    
    Example:
        >>> import os
        >>> print(os.getcwd())  # /home/user/project/subdir
        >>> path_to_parents(1)
        Changed working directory to: /home/user/project
        >>> path_to_parents(2)  # From /home/user/project
        Changed working directory to: /home/user
    
    Note:
        The function assumes that the specified number of parent directories exist.
        If not enough parent directories exist, the behavior is undefined.
    """
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)
    if levels > 1:
        for _ in range(levels-1):
            parent_dir = os.path.dirname(parent_dir)
            os.chdir(parent_dir)
    print(f"Changed working directory to: {parent_dir}")
    
    
def path_to_relative(relative_path):
    """
    Change the current working directory to a relative path.
    
    This function navigates to a new directory specified by a relative path from
    the current working directory. The function constructs the absolute path by
    joining the current directory with the provided relative path.
    
    Args:
        relative_path (str): The relative path to change the working directory to.
            This can be:
            - A subdirectory name (e.g., 'subdir')
            - A path with multiple directories (e.g., 'subdir/nested')
            - A path using '..' to go up levels (e.g., '../sibling')
            - Any valid relative path string
    
    Returns:
        None: This function modifies the current working directory in-place.
    
    Side Effects:
        - Changes the current working directory
        - Prints the new working directory path to stdout
    
    Example:
        >>> import os
        >>> print(os.getcwd())  # /home/user/project
        >>> path_to_relative('data')
        Changed working directory to: /home/user/project/data
        >>> path_to_relative('../scripts')
        Changed working directory to: /home/user/project/scripts
        >>> path_to_relative('models/trained')
        Changed working directory to: /home/user/project/scripts/models/trained
    
    Raises:
        OSError: If the specified relative path does not exist or is not accessible.
        NotADirectoryError: If the path exists but is not a directory.
    
    Note:
        The function does not validate if the target directory exists before attempting
        to change to it. If the directory doesn't exist, os.chdir() will raise an exception.
    """
    current_dir = os.getcwd()
    new_dir = os.path.join(current_dir, relative_path)
    os.chdir(new_dir)
    print(f"Changed working directory to: {new_dir}")
    
    
    
# def print_gpu_usage()
    
