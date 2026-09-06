from .atoms import AtomSet
from .bonds import BondSet
from .confs import ConfSet
from .mol import GraphMol, GraphBatch

# MDAnalysis.__init__ force-enables its own DeprecationWarnings with a ('once', ...) filter
# that overrides any 'ignore' filter we set. The deprecated topology.tables module is then
# imported during MDAnalysis init (and by prolif), so the warning fires before we can suppress
# it. Redirect stderr during the import to silence it.
import io
import contextlib
with contextlib.redirect_stderr(io.StringIO()):
    from .protein import Protein, ProteinBatch
    from .complex import BindingComplex, ComplexBatch
    from .interactions import Interaction, InteractionSet

from .vocab import Vocabulary, AtomVocab, BondVocab
