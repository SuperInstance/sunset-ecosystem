:- module fleet_12.

:- interface.
:- pred evaluate(float::out) is det.

:- implementation.
:- import_module float, list, string, bool.

evaluate(Result) :-
    Result = (1.0 + 2.0).
