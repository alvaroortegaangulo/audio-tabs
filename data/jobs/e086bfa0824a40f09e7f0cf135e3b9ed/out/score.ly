\version "2.24.0"

\paper {
  #(set-paper-size "letter")
  top-margin = 12\mm
  bottom-margin = 12\mm
  left-margin = 14\mm
  right-margin = 14\mm
  ragged-last = ##f
}

\header {
  title = "Knockin' On Heaven's Door [B0ds-417Odw] (mp3cut.net)"
  composer = ""
  tagline = ##f
}

global = {
  \numericTimeSignature
  \key g \major
  \time 4/4
  \tempo 4 = 70
}

chordsMusic = \chordmode {
  \set chordChanges = ##t
  gis8:7 g8 g8 g8 g8 d8 d8 d8 d8 d8 a8:m7 a8:m7 a8:m7 a8:m7 a8:m7 a8:m7 a8:m7 g8 g8 g8 g8 d8 d8 d8 d8 d8 c8 c8 c8 c8 c8 c8 e8:m7 e8:m7 g8 g8
}

rhythm = {
  \global
  \override Score.BarNumber.break-visibility = ##(#t #t #t)
  \override Score.BarNumber.font-size = #1
  \override Score.RehearsalMark.self-alignment-X = #LEFT
  \override Score.RehearsalMark.font-size = #2
  \improvisationOn
  \mark \markup \box "A" c4 c4 c4 c4 | c4 c4 c4 c4 | c4 c4 c4 c4 | c4 c4 c4 c4 | c4 c4 c4 c4 |
}

\layout {
  \context {
    \ChordNames
    \override ChordName.font-size = #2
  }
  \context {
    \Score
    \override SpacingSpanner.base-shortest-duration = #(ly:make-moment 1/16)
  }
}

\score {
  <<
    \new ChordNames \with { alignAboveContext = "staff" } { \chordsMusic }
    \new Staff = "staff" \with {
      instrumentName = "Gtr."
    } { \rhythm }
  >>
}
