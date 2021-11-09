:- ['four.pl'].

score([],0,0).
score([grade(N,O)|T],A,C):-score(T,M,K), A is M + O, C is K + 1.
average_score(P,S):-subject(P,L),score(L,A,C), S is A / C.

fail(N):-subject(P,L),member(grade(N,2),L),!.
fail_in_group(group(G,L),S):-member(S,L), fail(S).
fail_in_group_count(G,Count):-group(G,L),findall(S,fail_in_group(group(G,L),S), FL),length(FL, Count).

fail_in_subject_count(P,Count):-subject(P,L),findall(grade(N,2), member(grade(N,2),L), FL),length(FL, Count).
