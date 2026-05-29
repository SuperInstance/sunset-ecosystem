%% reasoning/mercury/reasoner.m — Mercury Reasoner for Plato
%%
%% Formal verification of tile reasoning properties.
%% Compile with: mmc --make reasoner
%% Test with: mercury-test

:- module reasoner.
:- interface.

:- import_module list.
:- import_module float.
:- import_module string.

%% Tile type: id, embedding (list of floats), score
:- type tile ---> tile(id :: int, embedding :: list(float), score :: float).

%% Compute cosine similarity between two embeddings
:- func cosine_similarity(list(float), list(float)) = float.
:- mode cosine_similarity(in, in) = out is det.

%% Find most similar tile in a list
:- pred find_most_similar(list(tile)::in, list(float)::in, tile::out, float::out) is semidet.

%% Safety property: all tiles have valid embeddings (non-empty, finite)
:- pred valid_tile(tile::in) is semidet.

%% Safety property: similarity is symmetric
:- pred similarity_symmetric(list(float)::in, list(float)::in) is semidet.

%% Safety property: similarity is bounded [0, 1]
:- pred similarity_bounded(list(float)::in, list(float)::in) is semidet.

%% Batch similarity computation
:- pred batch_similarity(list(float)::in, list(list(float))::in, list(float)::out) is det.

:- implementation.

:- import_module math.
:- import_module require.
:- import_module int.

% Dot product of two lists
dot_product([], []) = 0.0.
dot_product([H1 | T1], [H2 | T2]) = H1 * H2 + dot_product(T1, T2).
dot_product(_, _) = 0.0. % Different lengths

% Norm of a vector
norm([]) = 0.0.
norm([H | T]) = sqrt(H * H + norm(T) * norm(T)).

% Cosine similarity
cosine_similarity(A, B) = Result :-
    Dot = dot_product(A, B),
    NormA = norm(A),
    NormB = norm(B),
    ( if NormA * NormB = 0.0
      then Result = 0.0
      else Result = Dot / (NormA * NormB)
    ).

% Find most similar tile
find_most_similar([], _, _, _) :- fail.
find_most_similar([Tile | Tiles], Query, BestTile, BestScore) :-
    Score = cosine_similarity(Tile ^ embedding, Query),
    ( if Tiles = []
      then BestTile = Tile, BestScore = Score
      else find_most_similar(Tiles, Query, RestTile, RestScore),
           ( if Score > RestScore
             then BestTile = Tile, BestScore = Score
             else BestTile = RestTile, BestScore = RestScore
           )
    ).

% Valid tile: non-empty embedding, finite values
valid_tile(Tile) :-
    Tile ^ embedding \= [],
    all_finite(Tile ^ embedding).

:- pred all_finite(list(float)::in) is semidet.
all_finite([]).
all_finite([H | T]) :-
    H \= inf,
    H \= -inf,
    H \= nan,
    all_finite(T).

% Symmetry: cos_sim(A, B) = cos_sim(B, A)
similarity_symmetric(A, B) :-
    cosine_similarity(A, B) = cosine_similarity(B, A).

% Bounded: 0 <= cos_sim <= 1
similarity_bounded(A, B) :-
    Sim = cosine_similarity(A, B),
    Sim >= 0.0,
    Sim <= 1.0.

% Batch similarity: query against list of embeddings
batch_similarity(_, [], []).
batch_similarity(Query, [Emb | Rest], [Score | Scores]) :-
    Score = cosine_similarity(Query, Emb),
    batch_similarity(Query, Rest, Scores).

%% Test cases
:- pred test_similarity is det.
test_similarity :-
    A = [1.0, 0.0, 0.0],
    B = [1.0, 0.0, 0.0],
    Sim = cosine_similarity(A, B),
    require.unify(Sim, 1.0, "Identical vectors should have similarity 1.0").

:- pred test_orthogonal is det.
test_orthogonal :-
    A = [1.0, 0.0, 0.0],
    B = [0.0, 1.0, 0.0],
    Sim = cosine_similarity(A, B),
    require.unify(Sim, 0.0, "Orthogonal vectors should have similarity 0.0").

:- pred test_valid_tile is semidet.
test_valid_tile :-
    Tile = tile(1, [1.0, 2.0, 3.0], 0.0),
    valid_tile(Tile).

:- pred test_find_most_similar is semidet.
test_find_most_similar :-
    Tiles = [tile(1, [1.0, 0.0], 0.0),
             tile(2, [0.0, 1.0], 0.0),
             tile(3, [0.9, 0.1], 0.0)],
    Query = [1.0, 0.0],
    find_most_similar(Tiles, Query, Best, Score),
    Best ^ id = 1,
    Score > 0.99.

:- pred main is det.
main :-
    test_similarity,
    test_orthogonal,
    ( if test_valid_tile
      then io.write_string("valid_tile: OK\n")
      else io.write_string("valid_tile: FAIL\n")
    ),
    ( if test_find_most_similar
      then io.write_string("find_most_similar: OK\n")
      else io.write_string("find_most_similar: FAIL\n")
    ).
