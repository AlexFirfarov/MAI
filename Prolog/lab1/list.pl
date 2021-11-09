mylength([],0).
mylength([_|T],N):-mylength(T,M), N is M + 1.

mymember(X,[X|_]).
mymember(X,[_|T]):-mymember(X,T).

myappend([],X,X).
myappend([M|T],X,[M|Z]):-myappend(T,X,Z).

remove(X,[X|T],T).
remove(X,[M|T],[M|R]):-remove(X,T,R).

mypermute([],[]).
mypermute(L,[X|T]):-remove(X,L,R), mypermute(R,T).

mysublist(X,L):-myappend(_,T,L),myappend(X,_,T).

remove_n_1(0,L,L).
remove_n_1(N,[L|T],X):-remove(L,[L|T],K),R is N - 1,remove_n_1(R,K,X).

remove_n_2(0,L,L).
remove_n_2(N,[L|T],X):- R is N - 1, remove_n_2(R,T,X).

count_first_1([],0).
count_first_1([L|T],1):- not(member(L,T)).
count_first_1([L|T],N):-remove(L,T,X),count_first_1([L|X],M),N is M + 1,!.

count_element(_,[],0).
count_element(L,[L|T],N):-count_element(L,T,M), N is M + 1.
count_element(L,[_|T],N):-count_element(L,T,N).
count_first_2([],0).
count_first_2([L|T],N):-count_element(L,[L|T],N),!.

find_last_n(N,L,X):-mylength(L,K),O is K - N,remove_n_2(O,L,X). /*Найти последние n элементов*/
