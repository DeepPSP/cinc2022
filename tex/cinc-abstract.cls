%% cinc-abstract.cls 
%%
%% Adaptation of cinc.cls to suit the format of the conference program book
%% of "Computing in Cardiology"
%% George B. Moody (george@mit.edu)
%% 2 November 2009
%% * 1st revision  14 January 2014 (George Moody)
%%                  - uses 'ifpdf.sty' if available
%%                  - pdflatex /Creator tag is now CinC-abstract-2014
%%
%% Original cinc.cls by:
%% Hans A. Kestler {h.kestler@ieee.org OR hans.kestler@medizin.uni-ulm.de}
%% See cinc.cls for history
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{cinc-abstract}[2014/01/14]
\typeout{Document Class `cinc-abstract.cls' <14-January-2014>.}
\typeout{Adapted from cinc.cls by G.B. Moody}

\usepackage{graphicx}
\usepackage{wrapfig}
\IfFileExists{times.sty}{\usepackage{times}}{\@missingfileerror{times}{sty}}

% Detect PDF output mode (GBM, based on Heiko Oberdiek's ifpdf.sty)
\IfFileExists{ifpdf.sty}{\usepackage{ifpdf}}{
\newif\ifpdf
\ifx\pdfoutput\undefined
\else
 \ifx\pdfoutput\relax
 \else
  \ifcase\pdfoutput
  \else
   \pdftrue
  \fi
 \fi
\fi
}

\renewcommand{\normalsize}{\@normalsize}
\@options

\hoffset            -11mm
\voffset            -42mm
\topmargin           20mm
\oddsidemargin        0mm
\evensidemargin       0mm
\textheight         175mm
\textwidth          110mm
\columnsep            8mm
\parindent            1pc
\headsep              6mm
\headheight          12pt
\lineskip             1pt
\normallineskip       1pt
\partopsep            \z@
\topsep             1.3ex
\parsep               \z@
\itemsep              \z@

\def\baselinestretch{1}
\def\@normalsize{\@setsize\normalsize{12pt}\xpt\@xpt
\abovedisplayskip 1em plus2pt minus5pt\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip \z@ plus3pt\belowdisplayshortskip .6em plus3pt minus3pt
\topsep \belowdisplayshortskip
}
\def\small{\@setsize\small{9pt}\viiipt\@viiipt}
\def\footnotesize{\@setsize\footnotesize{9pt}\viiipt\@viiipt}
\def\scriptsize{\@setsize\scriptsize{8pt}\viipt\@viipt}
\def\tiny{\@setsize\tiny{6pt}\vpt\@vpt}
\def\large{\@setsize\large{14pt}\xiipt\@xiipt}
\def\Large{\@setsize\Large{18pt}\xivpt\@xivpt}
\def\LARGE{\@setsize\LARGE{22pt}\xviipt\@xviipt}
\def\huge{\@setsize\huge{22pt}\xxpt\@xxpt}
\def\Huge{\@setsize\Huge{28pt}\xxvpt\@xxvpt}

%% Definitions for PARAGRAPHS
\abovedisplayskip       0pt plus2pt minus5pt%
\belowdisplayskip       \abovedisplayskip
\abovedisplayshortskip  0pt plus3pt%
\belowdisplayshortskip  0pt plus3pt minus3pt%

%% Definitions for LISTS
\itemindent        -1em
\leftmargini        2em
\leftmarginii       1em
\leftmarginiii    1.5em
\leftmarginiv     1.5em
\leftmarginv      1.0em
\leftmarginvi     1.0em
\labelsep           5pt
\leftmargin\leftmargini
\labelwidth         \z@

\def\@listI{\leftmargin\leftmargini} \@listI
\def\@listi{\leftmargin\leftmargini  \topsep \z@ plus 1pt minus 1pt}
\def\@listii{\leftmargin\leftmarginii\labelwidth\leftmarginii
    \advance\labelwidth-\labelsep    \topsep \z@}
\def\@listiii{\leftmargin\leftmarginiii\labelwidth\leftmarginiii
    \advance\labelwidth-\labelsep    \topsep \z@}
\def\@listiv{\leftmargin\leftmarginiv\labelwidth\leftmarginiv
    \advance\labelwidth-\labelsep    \topsep \z@}
\def\@listv{\leftmargin\leftmarginv\labelwidth\leftmarginv
    \advance\labelwidth-\labelsep    \topsep \z@}
\def\@listvi{\leftmargin\leftmarginvi\labelwidth\leftmarginvi
    \advance\labelwidth-\labelsep    \topsep \z@}

\def\labelenumi{\theenumi.}     \def\theenumi{\arabic{enumi}}
\def\labelenumii{(\theenumii)}  \def\theenumii{\alph{enumii}}
\def\labelenumiii{\theenumiii.} \def\theenumiii{\roman{enumiii}}
\def\labelenumiv{\theenumiv.}   \def\theenumiv{\Alph{enumiv}}
\def\p@enumii{\theenumi}
\def\p@enumiii{\theenumi(\theenumii)}
\def\p@enumiv{\p@enumiii\theenumiii}

\def\labelitemi{$\scriptstyle\bullet$}
\def\labelitemii{\textbf{--}}
\def\labelitemiii{$\ast$}
\def\labelitemiv{$\cdot$}

%% \itemindent is set to \z@ by list, so define new temporary variable
\newdimen\tmpitemindent
\def\verse{\let\\=\@centercr
    \list{}{\itemsep\z@ \itemindent -1.5em \listparindent \itemindent
    \rightmargin\leftmargin\advance\leftmargin 1.5em}\item[]}
\let\endverse\endlist
\def\quotation{\list{}{\listparindent 1.5em \itemindent\listparindent
    \rightmargin\leftmargin \parsep 0pt plus 1pt}\item[]}
\let\endquotation=\endlist
\def\quote{\list{}{\rightmargin\leftmargin}\item[]}
\let\endquote=\endlist
\def\@mklab#1{#1}
\def\description{\tmpitemindent\itemindent\list{}{\itemindent\tmpitemindent
    \labelwidth\z@\def\makelabel##1{\hspace\labelsep\it{##1}}}}
\def\enddescription{\endlist\par}
\def\enumerate{\tmpitemindent\itemindent\ifnum \@enumdepth >3 \@toodeep\else
    \advance\@enumdepth \@ne \edef\@enumctr{enum\romannumeral\the\@enumdepth}%
    \list{\csname label\@enumctr\endcsname}{\itemindent\tmpitemindent
    \usecounter{\@enumctr}\def\makelabel##1{\hspace\labelsep\hfil{##1}}}\fi}
\def\endenumerate{\endlist\par}
\def\itemize{\tmpitemindent\itemindent\ifnum \@itemdepth >3 \@toodeep\else
    \advance\@itemdepth\@ne
    \edef\@itemitem{labelitem\romannumeral\the\@itemdepth}%
    \list{\csname\@itemitem\endcsname}{\itemindent\tmpitemindent
    \def\makelabel##1{\hspace\labelsep\hfil\it{##1}}}\fi}
\def\enditemize{\endlist\par}

\def\titlepage{ \newpage \thispagestyle{empty}\c@page\z@}
\def\endtitlepage{ \newpage }

\arraycolsep     5pt
\tabcolsep       6pt
\arrayrulewidth .4pt
\doublerulesep   2pt
\tabbingsep\labelsep

\def\theequation{\arabic{equation}}


%% Definitions for floats
\floatsep 12pt plus 2pt minus 2pt
\textfloatsep 20pt plus 2pt minus 4pt
\@fptop 0pt plus 1fil
\@fpsep 8pt plus 2fil
\@fpbot 0pt plus 1fil
\def\topfraction{1.0}
\def\floatpagefraction{0.8}
\setcounter{topnumber}{2}
\intextsep 12pt plus 2pt minus 2pt
\setcounter{bottomnumber}{2}
\def\bottomfraction{.4}
\setcounter{totalnumber}{4}
\def\textfraction{.2}


%% redefine CAPTION
\long\def\@caption#1[#2]#3{\par\addcontentsline{\csname ext@#1\endcsname}{#1}{%
   \protect\numberline{\csname the#1\endcsname}{\ignorespaces #2}}
   \begingroup \@parboxrestore \normalsize
     \@makecaption{\csname fnum@#1\endcsname}{\ignorespaces #3}{#1}\par
   \endgroup}

\def\tablestring{table}
\def\figurestring{figure}

\newlength{\captionindent}
\long\def\@makecaption#1#2#3{
\ifx\figurestring#3
    \vskip 5pt
    \setbox\@tempboxa\hbox{\normalsize #1.~ #2}
    \ifdim \wd\@tempboxa >\hsize
       \setbox\@tempboxa\hbox{\normalsize #1.~ }
       \setlength\captionindent{\wd\@tempboxa} \divide\captionindent by 2
       \parbox[t]{\hsize}{\normalsize
       \unhbox\@tempboxa#2}
    \else \hbox to\hsize{\small\hfil\box\@tempboxa\hfil}
    \fi
\else
    \vskip 5pt
    \setbox\@tempboxa\hbox{\normalsize #1.~ #2}
    \ifdim \wd\@tempboxa >\hsize
       \setbox\@tempboxa\hbox{\normalsize #1.~ }
       \setlength\captionindent{\wd\@tempboxa} \divide\captionindent by 2
       \parbox[t]{\hsize}{\normalsize
       \unhbox\@tempboxa#2}
    \else \hbox to\hsize{\small\hfil\box\@tempboxa\hfil}
    \fi
\fi}

\newcounter{figure}
\def\thefigure{\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{Figure~\thefigure}
\def\figure{\@float{figure}}
\let\endfigure\end@float

\newcounter{table}
\def\thetable{\@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{Table~\thetable}
\def\table{\@float{table}}
\let\endtable\end@float

\def\maketitle{\par
  \begingroup
  \newpage\global\@topnum\z@ \@maketitle
  \endgroup
  \let\maketitle\relax\let\@maketitle\relax
}

\def\@maketitle{
  \hyphenpenalty10000   % disable hyphenation in title block
  \raggedright
  \newpage
  {\large\textbf{\textsf\@title}}
  \vskip 3mm
  \textsf\@author
  \vskip 5mm
}

% Set parameters for PDF output if needed (GBM)
\ifpdf
\pdfpagewidth=5.5truein
\pdfpageheight=8.5truein
\pdfhorigin=1truein
\pdfvorigin=1truein
\pdfinfo{/Creator (CinC-abstract-2014)}
\fi

\endinput
