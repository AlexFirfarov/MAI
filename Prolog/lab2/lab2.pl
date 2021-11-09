remove(X,[X|T],T).
remove(X,[M|T],[M|R]):-remove(X,T,R).

mypermute([],[]).
mypermute(L,[X|T]):-remove(X,L,R), mypermute(R,T).

familys([['Aleksei Ivanovich',_],['Fedor Semenovich',_],['Valentin Petrovich',_],['Grigorii Arkadevich',_]]).
pairs([['Aleksei Ivanovich',_],['Fedor Semenovich',_],['Valentin Petrovich',_],['Grigorii Arkadevich',_]]).

rule1(_,Pair):-member(['Aleksei Ivanovich','Lenia'],Pair).
rule2(Fam,Pair):-member([Father,'Andrei'],Pair), member([Father,'Kolia'],Fam).
rule3(Fam,Pair):-member([Father,'Tima'],Pair), member([Father,'Andrei'],Fam).
rule4(Fam,Pair):-member(['Fedor Semenovich',Son],Pair), member(['Valentin Petrovich',Son],Fam).
rule5(Fam,Pair):-member(['Valentin Petrovich',Son],Pair), member(['Aleksei Ivanovich',Son],Fam).

rule([],Pair).
rule([FH|FT],Pair):- \+(member(FH,Pair)), rule(FT,Pair).

combine1(familys([['Aleksei Ivanovich',_],['Fedor Semenovich',_],['Valentin Petrovich',_],['Grigorii Arkadevich',_]]), [Z1,Z2,Z3,Z4], Result):- 
    append([], [['Aleksei Ivanovich',Z1],['Fedor Semenovich',Z2],['Valentin Petrovich',Z3],['Grigorii Arkadevich',Z4]], Result).

combine2(pairs([['Aleksei Ivanovich',_],['Fedor Semenovich',_],['Valentin Petrovich',_],['Grigorii Arkadevich',_]]), [Z1,Z2,Z3,Z4], Result):- 
    append([], [['Aleksei Ivanovich',Z1],['Fedor Semenovich',Z2],['Valentin Petrovich',Z3],['Grigorii Arkadevich',Z4]], Result).

write_list([]).
write_list([[X,Y]|T]):-
    write(X),
    write(' - '),
    write(Y),
    nl,
    write_list(T).

solve:-
    mypermute(['Lenia','Andrei','Kolia','Tima'],List1),
    combine1(familys([['Aleksei Ivanovich',_],['Fedor Semenovich',_],['Valentin Petrovich',_],['Grigorii Arkadevich',_]]), List1, Fam),
    mypermute(['Lenia','Andrei','Kolia','Tima'],List2),
    combine2(pairs([['Aleksei Ivanovich',_],['Fedor Semenovich',_],['Valentin Petrovich',_],['Grigorii Arkadevich',_]]), List2, Pair),
    rule1(Fam,Pair),
    rule2(Fam,Pair),
    rule3(Fam,Pair),
    rule4(Fam,Pair),
    rule5(Fam,Pair),
    rule(Fam,Pair),
    write('Father - Son'),
    nl,
    write_list(Fam),
    nl,
    write('Pairs'),
    nl,
    write_list(Pair).

 