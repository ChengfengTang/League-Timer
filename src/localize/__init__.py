"""Champion localization on screen (find *where* a tracked champion is).

The recognizer classifies *what* ability fires; the localizer answers *who and
where* by matching the champion's nameplate (e.g. "Ezreal") and cropping the
region below its healthbar. Works for your own champion (green bar) and enemies
(red bar) with the same template. This lets
the classifier run on a tight crop around the champion instead of the whole
frame, which removes teamfight noise, enlarges small VFX, and gives Flash a
spatial anchor.

The interface is multi-champion from the start: :meth:`Localizer.locate`
returns a list of :class:`Detection` (champion, box). The MVP wires up a single
champion (Ezreal), but adding more is just more name templates.
"""
from src.localize.localizer import Detection, Localizer

__all__ = ["Detection", "Localizer"]
