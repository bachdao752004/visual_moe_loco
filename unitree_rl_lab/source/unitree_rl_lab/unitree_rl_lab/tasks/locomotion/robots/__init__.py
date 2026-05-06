"""Robot task registrations for locomotion.

Current active scope registers Go2 agile only.
Other robot packages (g1/h1) are kept upstream and can be re-enabled explicitly.
"""

from . import go2

__all__ = ["go2"]
