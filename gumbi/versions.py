"""Parses current package version"""

import pkg_resources

try:
    __version__ = str(pkg_resources.get_distribution("gumbi").parsed_version)
except pkg_resources.DistributionNotFound:
    __version__ = "develop"