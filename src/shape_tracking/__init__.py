"""shape_tracking: catheter centerline shape tracking with ZED2 + NDI Aurora.

Import layout is deliberate: the top-level package pulls in only numpy + OpenCV
(board geometry and pose math), NOT the ZED SDK. Live capture lives in
``shape_tracking.zed_capture`` and is imported explicitly so that downstream
analysis projects can::

    from shape_tracking import boards, registration

without needing ``pyzed`` installed. Capture code requires the ``[capture]``
extra (the ZED python API).
"""

from . import boards, registration

__all__ = ["boards", "registration"]
__version__ = "0.1.0"
