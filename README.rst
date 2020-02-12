Markdown and reStructuredText
=============================
GitHub supports several lightweight markup languages for documentation;
the most popular ones (generally, not just at GitHub) are **Markdown**
and **reStructuredText**.  Markdown is sometimes considered easier to
use, and is often preferred when the purpose is simply to generate HTML.
On the other hand, reStructuredText is more extensible and powerful,
with native support (not just embedded HTML) for tables, as well as
things like automatic generation of tables of contents.

Unless you are embedding text into a programming language (especially
Python doc comments, for which purpose reStructuredText was originally
developed) you will probably choose to use Markdown for a small document
(like this Gist once was).  But as you add more sections and features, you may
decide that you want the power of reStructuredText.  To avoid having to
convert all of your Markdown formatting when that happens, use this Gist
as a guide to the markup syntax common to Markdown and reStructuredText.

By using only the common subset shared by these two lightweight markup
languages (largely from Setext, which inspired both of them) for your
documents, you can eliminate the need for conversion.  Even if you also
use features that are specific to one or the other, using the common
subset wherever possible allows for painless cut & paste between them.

Another advantage of restricting yourself to the common subset described
here is that the resulting source is more likely to be parsed the same way
by different Markdown processors, which may handle some details like
indentation slightly differently.

If you have already used Markdown-specific syntax, or you just don't
want to limit yourself to the (admittedly very limited) intersection of
functionality in the subset, you can use **Pandoc** to convert Markdown
into reStructuredText - the online converter at
<http://johnmacfarlane.net/pandoc/try> is handy if you don't happen to
have Haskell installed.  (In fact, you may find the extended Markdown
supported by Pandoc adds enough functionality that you don't need to use
reStructuredText at all - at least, once you get Haskell and Pandoc
installed.)


The Common Denominator
======================
The basics of text formatting are common to both, a blank line (it may
contain spaces or tabs, but nothing else) is used to separate paragraphs
and everything within a paragraph "flows" and is wrapped to the display
width.  Be sure to avoid initial indentation (either of an entire
paragraph or just the first line) for normal text, as indentation is
significant to both Markdown and reStructuredText, but in different
ways.

When writing text, you should try to wrap source lines to 72
characters - but only when initially entering them - avoid re-wrapping
text when you edit it later, to limit the number of "insignificant
changes" in your version control system (you *are* using one, right?!).
Feel free to use two spaces after the period at the end of a sentence
(or not) if that is useful for your text editor - I'm looking at you,
Emacs - since multiple spaces are treated the same as a single space.
Similarly, multiple blank lines are treated just like one blank line;
you may want to use multiple blank lines at the end of a section.


Font Faces - Emphasis and Examples
----------------------------------
Within paragraphs, inline markup provides basic formatting to add
emphasis to particular words or phrases, most commonly by making them
*italic* or **bold** (depending on the font, italics are often rendered
with an oblique face instead).  In technical writing, ``monospaced
text`` may be used to highlight program or function names, or for very
short snippets of code.

As with many types of formatting, Markdown provides multiple ways of
specifying font faces, one of which is also used by reStructuredText:
italics are marked with one asterisk (``*``) and bold face with two.  There
must be whitespace or other punctuation before the leading asterisks,
and after the trailing ones; furthermore there must be no space between
the asterisks and the text being emphasized.  Although Markdown supports
nesting bold and italic, reStructuredText does not (this is one of the
rare cases where Markdown is more powerful, as there is no way to
represent bold italics in reStructuredText).

Monospaced text is marked with two backquotes "``" instead of asterisks;
no bold or italic is possible within it (asterisks just represent
themselves), although in some contexts, code syntax highlighting may be
applied.  Note that in monospaced text, multiple spaces are *not*
collapsed, but are preserved; however, flow and wrapping *do* occur, and
any number of spaces may be replaced by a line break.  Markdown allows
monospaced text within bold or italic sections, but not vice versa -
reStructuredText allows neither.  In summary, the common inline markup
is the following::

    Mark *italic text* with one asterisk, **bold text** with two.
    For ``monospaced text``, use two "backquotes" instead.

Mark *italic text* with one asterisk, **bold text** with two.
For ``monospaced text``, use two "backquotes" instead.

-----

(Technically, Markdown uses just a single backquote for monospaced
text, however two backquotes work with most or all Markdown processors,
and are required for reStructuredText.)


Code or Console Example Blocks
------------------------------
If you have example code or console text that you want to appear with
all line breaks and relative indentation preserved, in a monospaced text
block, there is no common format for Markdown and reStructuredText, but
you can combine the formatting for both of them by ending one paragraph
with a double colon ``::`` (for reStructuredText) and indenting the next
one by four or more spaces (for Markdown) to make it appear in
monospaced font without flow or word-wrapping::

    A normal paragraph ending with ``::`` will flow and be word-wrapped::

        If the next paragraph is indented by four or more spaces, it will be monospaced text, without flow (or even wrapping in some non-print cases.)

        You can have multiple paragraphs like this, as long as they
        are all indented by the same amount.

A normal paragraph ending with ``::`` will flow and be word-wrapped::

    If the next paragraph is indented by four or more spaces, it will be monospaced text, without flow (or even wrapping in some non-print cases.)

    You can have multiple paragraphs like this, as long as they
    are all indented by the same amount.

-----

(We cheat a little bit here, Markdown does not interpret the
double-colon, but displays it as-is, whereas reStructuredText displays
just a single colon, but this is not too noticeable or annoying, as long
as you remember to use the double colon in your source.)


Line Blocks and Hard Line Breaks
--------------------------------
You may want to preserve line breaks in text blocks but don't want them
in monospaced text; common cases are verse (poetry or lyrics), street
addresses, and unadorned lists without bullets or numbering.  Markdown
and reStructuredText use completely different syntax for this, but you
can combine the markup for both reStructuredText line blocks and
Markdown hard line breaks by starting each line with a vertical bar (``|``)
and a space and ending it with two spaces.  For line breaks in the
source you don't want to preserve, omit the two spaces before the line
break and start the next line with one to three spaces instead.  Put a
blank line before the start and after the end of every line block.

Line blocks were added to reStructuredText in Docutils version 0.3.5
and there are some reStructuredText formatters that do not support
them; notably the GitHub README markup does not display them correctly.

These line blocks can also contain inline markup (which in a code
example block might be displayed literally), but keep any markup within
each line, since emphasis starting on one line and ending on another
applies to vertical bars between them, which appear in Markdown output::

    | *Yuku haru ya*  
    | *tori naki uo no*  
    | *me wa namida*  
    | -- **Matsuo Bashō**, The Narrow Road to Oku (*Oku no Hosomichi*),
     Tokyo, 1996, p. 23 (Translation: Donald Keene)
    | Spring is passing by!  
    | Birds are weeping and the eyes  
    | of fish fill with tears.  

| *Yuku haru ya*  
| *tori naki uo no*  
| *me wa namida*  
| -- **Matsuo Bashō**, The Narrow Road to Oku (*Oku no Hosomichi*),
 Tokyo, 1996, p. 23 (Translation: Donald Keene)  
| Spring is passing by!  
| Birds are weeping and the eyes  
| of fish fill with tears.  

------

(Again, we cheat a bit, since the Markdown output includes the vertical
bars; but at least they make it very clear when you end a line without
the required two spaces, something that is quite easy to do as there is
usually no visual indication of whether they are there or not.)


Block Quotations
----------------
When quoting long blocks of text from another writer, it is common
(especially in the context of e-mail) to set it off from the main text
by indenting it, possibly adding a vertical quotation line along the
left margin.  Markdown and reStructuredText use different syntax for
this, but you can combine their markup for block quotes by starting the
first line of a quotation with one space and a right angle bracket
(``>``), indenting all the remaining lines by one space as well (do not
add angle brackets to them).

Note that in reStructuredText, a block quotation cannot directly follow
a code example block - if it does it will be treated as part of the
example.  A normal paragraph or an "empty comment" (a line with only two
periods (``..``) and blank lines before and after it) must separate
them.

Every block quotation must have a blank line before and after it; they
can use the same inline markup as ordinary paragraphs.  Nested
quotations are possible by following a block quotation with another that
starts with two spaces and two right angle brackets; this allows up to
three levels of quotation (a fourth level is not possible since Markdown
treats four leading spaces as a code example).  While two right angle
brackets can be adjacent, three adjacent right angle brackets are a
doctest block (a special kind of code example block) in reStructuredText
and must have spaces between them to prevent that interpretation::

     > A block quotation will wrap and flow, and can have *inline*
     ``markup`` just like a regular paragraph.  It will be indented on
     the left (and possibly the right) margins, and may have a vertical
     quotation line on the left.

      >> With two spaces and two right angle brackets, a following block
      quotation will be doubly indented, and will have a second vertical
      quotation line along the left if quotation lines are generated.

       > >> A third level of quotation is the maximum level possible.

..

 > A block quotation will wrap and flow, and can have *inline*
 ``markup`` just like a regular paragraph.  It will be indented on
 the left (and possibly the right) margins, and may have a vertical
 quotation line on the left.

  >> With two spaces and two right angle brackets, a following block
  quotation will be doubly indented, and will have a second vertical
  quotation line along the left if quotation lines are generated.

   > >> A third level of quotation is the maximum level possible.

------

(The cheat here is that the reStructuredText output includes the right
angle bracket(s) on the first line in addition to the indentation; this
is not ideal, but is generally acceptable when used for a quotation, and
not just indented text.)


Titles and Section headers
--------------------------
Both Markdown and reStructuredText allow you to structure your document
by adding header titles for sections and subsections.  While they each
support a large number of levels of headers in different ways, the
common subset only has two levels: titles, formed by underlining the
text with ``==``, and subtitles, formed by underlining with ``--``.  The
underlining must be on the very next line, and be at least
as long as the (sub)title::

    Section Title
    =============
    The Common Denominator
    ======================

    Subsection Subtitle
    -------------------
    Titles and Section headers
    --------------------------

Note that a blank line after the underlining is optional, but a blank
line before the (sub)title is required.


Bulleted and Enumerated Lists
-----------------------------
In addition to (sub)section headers, both Markdown and reStructuredText
support itemized lists; these can be numbered (enumerated) or unnumbered
(bulleted) and the two types of lists can be nested within themselves
and each other.  List items are lines starting (possibly after spaces
for indentation) with a bullet symbol (``*``, ``-``, or ``+``) for bulleted
lists, or a number and a period (``1.``) for enumerated lists; in both
cases followed by one or more spaces and then the item text.  Although
reStructuredText supports other symbols for bulleted lists and
parentheses instead of period for enumerated lists, as well as ``#`` in
place of the number for auto-enumeration, Markdown only supports the
subset described above.

The spaces after the symbol or number determine the indentation needed
for additional item text on continuation lines or following paragraphs,
as well as the symbols or numbers for sub-lists.  Symbol or number indentation
of all items at any nesting level must be the same (even for long
enumerated lists with two-digit numbers) but the indentation of the text
of different items need not be the same.

If a list item contains multiple paragraphs (separated by blank lines)
or sub-lists, the indentation of the item text must be at least four
spaces more than the item symbol or number; this usually requires extra
spaces between the number and period (or symbol) and the item text.

A blank line is required before the first item and after the last item
in every top-level list, but is optional between items.  A blank line
is also required by reStructuredText before the first item of a
sub-list; omitting it sometimes appears to work, but only because the
sub-list is indented more than the item text.  This extra indentation
may cause the item text to be treated as part of a definition list and
displayed in bold; in other cases, it causes the sub-list to be
wrapped within a block quote, causing both the left and right margins
to move inwards and creating a double-indent effect.

A sub-list without a preceding blank line can also work if there is no
item text preceding the sub-list; but this generates odd-looking
output that is confusing to human readers, with the first bullet or
number of the sub-list on the same line as the bullet or number of the
item in the enclosing list.

While Markdown does not require a blank line before a sub-list, a blank line
between items changes the inter-item spacing (typically by creating
``<p>`` paragraph tags).  For consistent results, do not use blank lines
between items unless you must (for sub-lists), in which case use blank
lines between *all* the items at the same level (sub-list items do not
require the blank lines unless there are sub-sub-lists).

Markdown ignores the actual numbers given for enumerated lists and
always renumbers them starting with 1, but reStructuredText requires
that the numbers be in sequential order; the number of the first item
may or may not be preserved.  For compatibility, always start enumerated
lists with 1 and number them sequentially.  You should never mix
enumerated and bulleted items (or different bullet symbols) at the same
level; reStructuredText will reject it with an error (or, if there is a
blank line between them, create a new list).  On the other hand,
Markdown processors will combine adjacent bulleted and enumerated lists
(using the formatting of the first list); to create separate lists it is
not enough to have a blank line, there must be a non-list paragraph
between them.

Because Markdown formatting requires additional indentation for extra
paragraphs of item text in lists, the approach for monospaced paragraphs
given above in *Code or Console Example Blocks* requires additional
indentation of at least **eight** spaces (not just four) for example
blocks in lists.

Finally, it is a *very* good idea to make sure that your document
source does not contain any tab characters, especially when working
with multiple levels of sub-lists.  Configure your text editor to
expand all tabs into spaces; this will help to ensure that the initial
indentation is consistent and avoid errors if another editor
interprets the tabs differently.

The following two lists summarize and provide examples of the rules for
lists compatible with Markdown and reStructuredText::

    *   Mark bulleted lists with one of three symbols followed by a space:

        1. asterisk (``*``)
        2. hyphen (``-``)
        3. plus sign (``+``)

    * Mark enumerated lists with a number, period (``.``) and a space.

    * The choice of symbol does not affect the output bullet style,
      which is solely determined by nesting level.
      Items can be continued on following lines indented at the same
      level as the item text on the first line, and will flow and wrap
      normally.

    *   The source indentation of item text can vary for different items
        (but continuation lines must be indented by the same amount as
        the item text that they are continuing).

        Additional paragraphs of item text (after a blank line) also
        require this indentation, with the extra requirement that it be
        four to seven spaces more than the item symbol or number.

        * These indentation requirements are the same for sub-list items
          (but apply to their symbol or number, not their item text).

    *   Blank lines between list items are optional, avoid them.

        + If you *do* use them (for items with sub-lists or extra
          paragraphs) put blank lines between *all* items at that level.

    A non-list paragraph is required to separate adjacent enumerated and
    bulleted lists, to keep Markdown from merging the second one into the
    first (and using the first style for both).

    1. Always put a blank line before the start of a list or sub-list.

    2. Use the same bullet symbol for all items in a bulleted list.

    3. Always start enumerated lists with 1.

    4. Use sequentially increasing numbers for succeeding list items.

    5.  Do not mix numbers and/or different bullet symbols at one level

        * (but this is okay for different levels or separate sublists).

    6.  Indent sub-lists by the same amount as the item text;
        this must be 4-7 spaces more than the symbol or number.

        1.  if enumerated, always start them with 1.

            + (the same rules apply to sub-sub-lists, etcetera)

        Additional non-sub-list paragraphs require the same indentation;
        example blocks (after double colon ``::``) must be indented at
        least eight spaces more than the symbol or number, like this::

            * item text::

                    code block

    7.  Indent symbols or numbers the same amount for any one list level.

        - (top-level list items should not have any leading indentation)

    8.  Align two-digit enumerated items by first digit, not the period.

    9.  Don't put leading zeros on enumerated items to align the periods

        * (use spaces after period if you want to align the item text in source).

    10. Make sure there are no tab characters in initial indentation.

    11. Always put a blank line after the end of a (top-level) list.

*   Mark bulleted lists with one of three symbols followed by a space:

    1. asterisk (``*``)
    2. hyphen (``-``)
    3. plus sign (``+``)

* Mark enumerated lists with a number, period (``.``) and a space.

* The choice of symbol does not affect the output bullet style,
  which is solely determined by nesting level.
  Items can be continued on following lines indented at the same
  level as the item text on the first line, and will flow and wrap
  normally.

*   The source indentation of item text can vary for different items
    (but continuation lines must be indented by the same amount as
    the item text that they are continuing).

    Additional paragraphs of item text (after a blank line) also
    require this indentation, with the extra requirement that it be
    four to seven spaces more than the item symbol or number.

    * These indentation requirements are the same for sub-list items
      (but apply to their symbol or number, not their item text).

*   Blank lines between list items are optional, avoid them.

    + If you *do* use them (for items with sub-lists or extra
      paragraphs) put blank lines between *all* items at that level.

A non-list paragraph is required to separate adjacent enumerated and
bulleted lists, to keep Markdown from merging the second one into the
first (and using the first style for both).

1. Always put a blank line before the start of a list or sub-list.

2. Use the same bullet symbol for all items in a bulleted list.

3. Always start enumerated lists with 1.

4. Use sequentially increasing numbers for succeeding list items.

5.  Do not mix numbers and/or different bullet symbols at one level

    * (but this is okay for different levels or separate sublists).

6.  Indent sub-lists by the same amount as the item text;
    this must be 4-7 spaces more than the symbol or number.

    1.  if enumerated, always start them with 1.

        + (the same rules apply to sub-sub-lists, etcetera)

    Additional non-sub-list paragraphs require the same indentation;
    example blocks (after double colon ``::``) must be indented at
    least eight spaces more than the symbol or number, like this::

        * item text::

                code block

7.  Indent symbols or numbers the same amount for any one list level.

    - (top-level list items should not have any leading indentation)

8.  Align two-digit enumerated items by first digit, not the period.

9.  Don't put leading zeros on enumerated items to align the periods

    * (use spaces after period if you want to align the item text in source).

10. Make sure there are no tab characters in initial indentation.

11. Always put a blank line after the end of a (top-level) list.


Hyperlink URLs
--------------
Markdown and reStructuredText use different and incompatible syntax for
arbitrary text hyperlinks, but reStructuredText will generate hyperlinks
for e-mail addresses or URLs, and Markdown will do so as well if they
are enclosed in angle brackets (``<>``).  Some Markdown processors do
not require the angle brackets, but there is little reason to omit them,
as they hardly affect readability, and explicitly specify whether or not
punctuation at the end of the URL is really part of the link.  Even
relative URLs can be used if the protocol is explicitly specified::

    The latest version of this document can be found at
    <https://gist.github.com/1855764>; if you are viewing it there (via
    HTTPS), you can download the Markdown/reStructuredText source at
    <https:/gists/1855764/download>.  You can contact the author via
    e-mail at <alex.dupuy@mac.com>.

The latest version of this document can be found at
<https://gist.github.com/1855764>; if you are viewing it there (via
HTTPS), you can download the Markdown/reStructuredText source at
<https:/gists/1855764/download>.  You can contact the author via
e-mail at <alex.dupuy@mac.com>.

-----

(Using the URLs directly for hyperlinks also means that even if a
Markdown processor has link generation disabled, a human reader can
always copy and paste the URL.)


Horizontal Rules (Transitions)
------------------------------
You can create a horizontal rule (a "transition" in reStructuredText
terminology) by placing four or more hyphens (``-``), asterisks (``*``),
or underscores (``_``) on a line by themselves, with blank lines before
and after and no indentation (trailing spaces are okay, but not leading
spaces).  Although Markdown requires only three, and allows spaces
between them, reStructuredText requires four repeated punctuation
characters.  Also, reStructuredText requires paragraphs before and after
the transition (code blocks or enumerated/bulleted list items are okay,
but section headers are not).

-----

Each of the following lines will produce a horizontal rule like the
one above::

    ****
    ______
    ----------


Not-Incompatible Extensions
===========================
Both Markdown and reStructuredText have markup that is not interpreted
by the other (either in the same or in an incompatible way), and which
is not too painful to read when rendered as ordinary text.  Hyperlink
URLs (as noted above) fall into this category for some basic Markdown
implementations that do not implement URL recognition.


Tables
------
Markdown has no support for tables (one of its biggest weaknesses); to
create them requires embedded HTML (if that is even allowed).  However,
the reStructuredText table format is fairly readable in original source
form (basic monospaced ASCII layout) so if you indent reStructuredText
tables by four or more spaces (and make sure that the previous paragraph
does *not* end with a double colon ``::``) you will get a nicely
formatted table in reStructuredText and a readable ASCII table in
Markdown.  There are two flavors of table markup in reStructuredText,
grid tables and simple tables.  Grid tables are trickier to generate, but
more flexible, and look nicer in source format::

    Make sure previous paragraph does not end with ``::``.

    +-------+----------+------+
    | Table Headings   | Here |
    +-------+----------+------+
    | Sub   | Headings | Too  |
    +=======+==========+======+
    | cell  | column spanning |
    + spans +----------+------+
    | rows  | normal   | cell |
    +-------+----------+------+
    | multi | * cells can be  |
    | line  | * formatted     |
    | cells | * paragraphs    |
    | too   |                 |
    +-------+-----------------+

Make sure previous paragraph does not end with ``::``.

    +-------+----------+------+
    | Table Headings   | Here |
    +-------+----------+------+
    | Sub   | Headings | Too  |
    +=======+==========+======+
    | cell  | column spanning |
    + spans +----------+------+
    | rows  | normal   | cell |
    +-------+----------+------+
    | multi | * cells can be  |
    | line  | * formatted     |
    | cells | * paragraphs    |
    | too   |                 |
    +-------+-----------------+

-----

A significant advantage of grid tables is that Pandoc Markdown supports
them, which is *not* the case for simple tables, for which Pandoc uses a
somewhat similar but incompatible format.  However, for Pandoc to
actually process the formatting, the four space indentation of the grid
tables must be removed (to prevent monospaced code block formatting).

Simple tables are easier, but cells must be on a single line and cannot
span rows::

    ===== ========= =====
    Table Headings  Here
    --------------- -----
    Sub   Headings  Too
    ===== ========= =====
    column spanning no
    --------------- -----
    cell  cell      row
    column spanning spans
    =============== =====

Note that lines between rows are optional and only needed to indicate
where cells in the previous line span columns (by omitting the space).

    ===== ========= =====
    Table Headings  Here
    --------------- -----
    Sub   Headings  Too
    ===== ========= =====
    column spanning no
    --------------- -----
    cell  cell      row
    column spanning spans
    =============== =====

-----

Apart from the ability to span rows and do block formatting within cells
in a grid table, the actual table formatting is not affected by the use
of grid or simple tables, and depends only on the reStructuredText
processor and any style sheets it may use; for more visual compatibility
you may want to use the table style that most closely resembles the
output table.

Also, just as for list indentation, it is a *very* good idea to make
sure that no tab characters are embedded in the tables; configure your
text editor to expand all tabs into spaces; this will help to ensure
that the source ASCII display in Markdown is properly aligned.


Comments
--------
There is no comment syntax for Markdown, but HTML comments can be used
with Markdown processors that allow them (raw HTML is often disabled
for security or other reasons, possibly with whitelisted tags allowed;
notably, GitHub and BitBucket README markdown disable HTML comments).
Standard Markdown (but not most processors) requires blank lines before
and after HTML blocks.  Comments in reStructuredText use a different
syntax, but it is possible to create comments that are entirely
invisible in reStructuredText output, and only appear as periods in
Markdown output (unless HTML comments are disabled).

In the following comment examples, the reStructuredText comment /
directive marker ``.. `` is followed by two more periods so that the
following blank line does not terminate the comment.  For most Markdown
processors, you can use an ``&nbsp;`` entity instead of the two
additional periods to reduce the visual impact; but some Markdown
processors (notably the Python Markdown used by BitBucket README
processing) do not support entities outside of HTML blocks.

The following block is completely hidden from reStructuredText output,
and barely visible in Markdown output if HTML comments are allowed::

    .. ..

     <!--- Need blank line before this line (and the .. line above).
     HTML comment written with 3 dashes so that Pandoc suppresses it.
     Blank lines may appear anywhere in the comment.

     All non-blank lines must be indented at least one space.
     HTML comment close must be followed by a blank line and a line
     that is not indented at all (if necessary that can be a line
     with just two periods followed by another blank line).
     --->

.. ..

 <!--- Need blank line before this line (and the .. line above).
 HTML comment written with 3 dashes so that Pandoc suppresses it.
 Blank lines may appear anywhere in the comment.

 All non-blank lines must be indented at least one space.
 HTML comment close must be followed by a blank line and a line
 that is not indented at all (if necessary that can be a line
 with just two periods followed by another blank line).
 --->

-----

You can also use a variation of the above to include Markdown markup
that will be entirely ignored by reStructuredText::

    .. ..

     <ul><li>Need blank line before this line (and .. line above).</li>
     <li>Blank lines may appear anywhere in this section.</li>

     <li>All non-blank lines must be indented at least one space.</li>
     <li>HTML and text are displayed only in Markdown output.</li></ul>
     <p>End of Markdown-only input must be followed by a blank line and
     a line that is not indented at all (if necessary that can be a line
     with just two dots followed by another blank line).</p>

.. ..

 <ul><li>Need blank line before this line (and .. line above).</li>
 <li>Blank lines may appear anywhere in this section.</li>

 <li>All non-blank lines must be indented at least one space.</li>
 <li>HTML and text are displayed only in Markdown output.</li></ul>
 <p>End of Markdown-only input must be followed by a blank line and
 a line that is not indented at all (if necessary that can be a line
 with just two dots followed by another blank line).</p>

-----

You can use another variation of the above to include reStructuredText
markup that will be ignored by Markdown (except for the periods)::

    .. ..

     <!--- Need blank line before this line (and the .. line above).
     HTML comment written with 3 dashes so that Pandoc suppresses it.
     These lines not starting with .. must be indented.
     HTML comment close must be followed by a blank line and a line
     that is not indented at all (if necessary that can be a line
     with just two periods followed by another blank line).

    .. note:: This is a reStructuredText directive - the Markdown
       output should be just periods

    .. --->

.. ..

 <!--- Need blank line before this line (and the .. line above).
 HTML comment written with 3 dashes so that Pandoc suppresses it.
 These lines not starting with .. must be indented.
 HTML comment close must be followed by a blank line and a line
 that is not indented at all (if necessary that can be a line
 with just two periods followed by another blank line).

.. note:: This is a reStructuredText directive - the Markdown
   output should be just periods

.. --->

-----

Note that although HTML comments are usually marked with ``<!-- -->``
you should use three dashes instead of two: ``<!--- --->`` as this is
used by Pandoc to prevent passing the comment through to the output.


Markdown Extensions
===================
Unlike reStructuredText, which is virtually identical across all its
implementations, there are a wide variety of semi-compatible Markdown
extension styles; the most popular are MultiMarkdown and Markdown Extra
(the latter implemented by PHP Markdown and Maruku, and partially by
Python Markdown and Redcarpet); Pandoc has its own set of Markdown
extensions, based on both Markdown Extra and reStructuredText; these
Markdown extensions are the most similar to reStructuredText, while the
Markdown Extra extensions have a smaller overlap, and the MultiMarkdown
extensions are only compatible with reStructuredText when they are also
identical to parts of Markdown Extra.

Definition Lists
----------------
Markdown Extra, MultiMarkdown, and Pandoc support a syntax that is
fairly compatible with the definition list syntax in reStructuredText;
by using the following format, definitions can be written that are
recognized by all of these processors.  In reStructuredText, any line
that is followed immediately by an indented line is a definition term,
with the following lines at the same indentation level forming the
definition.  Markdown Extra allows an optional blank line between the
term and definition lines, but requires the definition to begin with a
colon (``:``) that is not indented by more than three spaces and is
followed by a space and the definition

To be recognized as a definition list item in both reStructuredText and
Markdown extensions, only a single term is allowed, and it must be
followed immediately (with no blank line) by the definition.  The
definition must begin with an (indented) colon and a space and it and
any continuation lines or additional paragraphs or definitions must all
be indented by the same amount (one to three spaces), as shown in the
following example::

    term
      : definition

    longer term
      : multi-line definition
      a second line (will be subject to flow and wrapping)

      a second paragraph in the definition

    complex term
      : first definition

      : second definition

term
  : definition

longer term
  : multi-line definition
  a second line (will be subject to flow and wrapping)

  a second paragraph in the definition

complex term
  : first definition

  : second definition


Fancy list numbers
------------------
Although most Markdown processors only support enumerated lists with
arabic numbers followed by periods, Pandoc also supports other list
styles that are compatible with reStructuredText.  In particular,
letters (``A``) as well as roman numerals (``IV``) and alternate
punctuation with parentheses ( ``(b)`` or ``xiv)`` ) are recognized, and
sequences starting with numbers other than 1 (or roman numeral I or
letter A) have the actual starting number or letter preserved in output.
