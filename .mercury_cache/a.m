:- module fleet_a1b1.

:- interface.
:- pred evaluate(float::out) is det.

:- implementation.
:- import_module float, list, string, bool.

evaluate(Result) :-
    Result = (a1 + b1).
