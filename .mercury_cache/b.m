:- module fleet_2.

:- interface.
:- pred evaluate(float::out) is det.

:- implementation.
:- import_module float, list, string, bool.

evaluate(Result) :-
    Result = 2.0.
