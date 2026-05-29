:- module fleet_iffleethealth05passfail.

:- interface.
:- pred evaluate(float::out) is det.

:- implementation.
:- import_module float, list, string, bool.

evaluate(Result) :-
    Result = (if (fleet_health_value() > 0.5) then pass else fail).
