toar([H|[]],H):-!.
toar([Num1,'+',Num2|Tail],Res):-
     Res1 = Num1 + Num2,
	 List = [Res1|Tail],!,
     toar(List,Res).
toar([Num1,'-',Num2|Tail],Res):-
     Res1 = Num1 - Num2,
	 List = [Res1|Tail],!,
     toar(List,Res).
toar([N1 + N2,'*',Num2|Tail],Res):-
     Res1 = N1 + N2 * Num2,
	 List = [Res1|Tail],!,
     toar(List,Res).
toar([N1 - N2,'*',Num2|Tail],Res):-
     Res1 = N1 - N2 * Num2,
	 List = [Res1|Tail],!,
     toar(List,Res).
toar([Num1,'*',Num2|Tail],Res):-
     Res1 = Num1 * Num2,
	 List = [Res1|Tail],!,
     toar(List,Res).
toar([N1 + N2,'/',Num2|Tail],Res):-
     Res1 = N1 + N2 / Num2,
	 List = [Res1|Tail],!,
     toar(List,Res).
toar([N1 - N2,'/',Num2|Tail],Res):-
     Res1 = N1 - N2 / Num2,
	 List = [Res1|Tail],!,
     toar(List,Res).
toar([Num1,'/',Num2|Tail],Res):-
     Res1 = Num1 / Num2,
	 List = [Res1|Tail],!,
     toar(List,Res).

calculate(Expr,Result):-
     toar(Expr,Arifm),
     pref(Arifm,List),
     lin(List,Result).
    
pref(Expr,Expr):-atomic(Expr).
pref(Expr,Res):-
    Expr=..[Op,A,B],
    pref(A,A1),
    pref(B,B1),
    Res=[Op,A1,B1].

lin([H|T],List):-
    lin(H,H1),
    lin(T,T1),
    append(H1,T1,List),!.
lin([],[]):-!.
lin(+,["+"]):-!.
lin(-,["-"]):-!.
lin(*,["*"]):-!.
lin(/,["/"]):-!.
lin(X,[X]):-!.