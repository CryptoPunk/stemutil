import mutagen
import mutagen.mp4

fn = '../stems/Massive Attack/Mezzanine/01 - Massive Attack - Angel.stem.mp4'

filething_ctx = mutagen._util._openfile(instance=None,filething=None,fileobj=None,filename=fn,writable=False,create=False)

#Get Stem Data

with filething_ctx as filething:
    atoms = mutagen.mp4._atom.Atoms(filething.fileobj)
    stem = atoms.path(b'moov',b'udta',b'stem')[-1]
    trak = atoms.path(b'moov',b'trak')[-1]
    tkhds = [ next(t.findall(b'tkhd')) for t in atoms.path(b'moov')[-1].findall(b'trak') ] 
    tkhds = [ (t, t.read(filething.fileobj)) for t in tkhds ]
    # Set bit 32 to 0 to disable track
    # ref: https://developer.apple.com/library/archive/documentation/QuickTime/QTFF/QTFFChap2/qtff2.html#//apple_ref/doc/uid/TP40000939-CH204-33299
    ftyp = atoms.path(b'ftyp')[0]
    ftyp_data = ftyp.read(filething.fileobj)
    data = stem.read(filething.fileobj)

# Back to file
'''
stem = Atom.render(b'stem',data[1])

try:
    path = atoms.path(b"moov", b"udta")
except KeyError:
    path = atoms.path(b"moov")

if path[-1].name != b"udta":
    # moov.udta not found -- create one
    data = Atom.render(b"udta",stem)

# To finish saving take note of how room is made in the __save, __save_new, and __save_existing functions of MP4Tags


Also set
major brand to M4A:0
remove brands
iso2	MP4 Base Media v2 [ISO 14496-12:2005]
isom	MP4  Base Media v1 [IS0 14496-12:2003]
'''

