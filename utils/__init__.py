"""
Utility functions module for OPView.

Provides project scanning, VTK file management, and path resolution utilities.
"""

from .project_scanner import (
    scan_project_folders,
    get_project_folder_options,
)

from .vtk_utils import (
    get_reader,
    reader_cache,
    list_vtk_files,
    list_comparison_files,
    latest_file,
    ReaderCache,
)

from .path_utils import (
    resolve_vtk_path,
)

__all__ = [
    # Project scanner
    'scan_project_folders',
    'get_project_folder_options',
    
    # VTK utilities
    'get_reader',
    'reader_cache',
    'list_vtk_files',
    'list_comparison_files',
    'latest_file',
    'ReaderCache',
    
    # Path utilities
    'resolve_vtk_path',
]
