move(s(L),s(NewL)):-
    length(L,Length),
    Length1 is Length - 1,
    between(0,Length1,X),
    between(0,Length1,Y),
    X < Y,
    swap(L,X,Y,NewL).

prolong([X|T],[Y,X|T]):-
    move(X,Y),
    \+member(Y,[X|T]).

swap(L,X,Y,NewL):-
    length(L,Len1),
    length(NewL,Len2),
    Len1 =:= Len2,
    append(MH,[M|MT],L),
    length(MH,X),
    append(MH,[N|MT],Temp),
    append(NH,[N|NT],Temp),
    length(NH,Y),
    append(NH,[M|NT],NewL),
    !.

path_dfs(X,Y,P):-dfs([X],Y,P).
dfs([X|T],X,[X|T]).
dfs(P,Y,R):-
    prolong(P,P1),
    dfs(P1,Y,R).

path_bdth(X,Y,P):-bdth([[X]],Y,P).
bdth([[X|T]|_],X,[X|T]).
bdth([P|QI],X,R):-
    findall(Z,prolong(P,Z),T),
    append(QI,T,QO),
    !,
    bdth(QO,X,R).
bdth([_|T],Y,L):-bdth(T,Y,L).

search_id(Start,Finish,Path):-
    integer_1(Level),
    search_id(Start,Finish,Path,Level).

integer_1(1).
integer_1(M):-integer_1(N), M is N + 1.

search_id(Start,Finish,Path,DepthLimit):-
    depth_id([Start],Finish,Path,DepthLimit).

depth_id([Finish|T],Finish,[Finish|T],_).
depth_id(Path,Finish,R,N):-
    N > 0,
    prolong(Path,NewPath),
    N1 is N - 1,
    depth_id(NewPath,Finish,R,N1).
    
search_dfs(Start,Goal):-
    write('DFS'),
    nl,
    get_time(Time),
    path_dfs(Start,Goal,List),
    print_path(List),
    get_time(Time1),
    write('DFS END'),
    nl,
    T is Time1 - Time,
    write('Time: '), write(T),nl,nl.

search_bdth(Start,Goal):-
    write('BDTH'),
    nl,
    get_time(Time),
    path_bdth(Start,Goal,List),
    print_path(List),
    get_time(Time1),
    write('BDTH END'),
    nl,
    T is Time1 - Time,
    write('Time: '), write(T),nl,nl.

search_id(Start,Goal):-
    write('ITER'),
    nl,
    get_time(Time),
    search_id(Start,Goal,List),
    print_path(List),
    get_time(Time1),
    write('ITER END'),
    nl,
    T is Time1 - Time,
    write('Time: '), write(T),nl,nl.
    
print_path([]).
print_path([H|T]):-print_path(T), write(H), nl.
