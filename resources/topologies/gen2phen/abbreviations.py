#!/usr/bin/env python
'''Link abbreviations to their full names

Based on

A Simple Algorithm for Identifying Abbreviations Definitions in Biomedical Text
A. Schwartz and M. Hearst
Biocomputing, 2003, pp 451-462.


# License: GNU General Public License, see http://www.clips.ua.ac.be/~vincent/scripts/LICENSE.txt
'''
__date__ = 'July 2012'
__author__ = 'Vincent Van Asch'
__version__ = '1.2.1'

__doc__='''
This script takes tokenized sentences and prints the abbreviations and their definitions to STDOUT.


REMARKS
    The algorithm detects only links between an abbreviation and its definition if they
    are in the format <definition> (<abbreviation>). So, the reverse
    <abbreviation> (<definition>) is not detected.

    The algorithm can only find definitions that have all characters of the abbreviation
    in them.

    It will also make errors in cases like "tyrosine transcription factor I (TTFI)"
    the wrong definition will be "transcription factor I" because all characters from
    the abbreviation are in the definition.

    On the labeled yeast corpus (http://biotext.berkeley.edu/data.html) version 1.0 of this
    script reaches:

    TP: 673
    FP: 94
    FN: 280

    P : 87.74
    R : 70.62
    F1: 78.26

    (The corpus had to be corrected manually in order to be useable.)

ACKNOWLEDGEMENTS
    Based on:
    A Simple Algorithm for Identifying Abbreviations Definitions in Biomedical Text
    A. Schwartz and M. Hearst
    Biocomputing, 2003, pp 451-462.

%s (version %s)''' %(__date__, __version__)

__REPRODUCE__ = False

import os, sys, re, getopt, logging

log = logging.getLogger(__name__)

encoding='UTF8'

class Candidate(unicode):
    def __new__(cls, start, stop, str):
        return unicode.__new__(cls, str)
    def __init__(self, start, stop, str):
        self._start = start
        self._stop = stop

    def __getslice__(self, i, j):
        start = self.start+i
        stop = self.start+j
        str = unicode.__getslice__(self, i, j)
        return Candidate(start, stop, str)

    @property
    def start(self):
        '''The start index'''
        return self._start
    @property
    def stop(self):
        '''The stop index'''
        return self._stop


def getcandidates(sentence):
    '''Yields Candidates'''
    if '(' in sentence:
        # Check some things first
        if sentence.count('(') != sentence.count(')'):
            raise ValueError('Unbalanced parentheses: %s' %sentence)

        if sentence.find('(') > sentence.find(')'):
            raise ValueError('First parentheses is right: %s' %sentence)

        closeindex = -1
        while 1:
            # Look for open parenthesis
            openindex = sentence.find('(', closeindex+1)

            if openindex == -1: break

            # Look for closing parantheses
            closeindex = openindex+1
            open=1
            skip=False
            while open:
                try:
                    char = sentence[closeindex]
                except IndexError:
                    # We found an opening bracket but no associated closing bracket
                    # Skip the opening bracket
                    skip=True
                    break
                if char == '(':
                    open +=1
                elif char == ')':
                    open -=1
                closeindex+=1

            if skip:
                closeindex = openindex+1
                continue

            # Output if conditions are met
            start = openindex+1
            stop = closeindex-1
            str = sentence[start:stop]

            # Take into account whitepsace that should be removed
            start = start + len(str) - len(str.lstrip())
            stop = stop - len(str) + len(str.rstrip())
            str = sentence[start:stop]

            if conditions(str):
                yield Candidate(start, stop, str)






def conditions(str):
    '''Based on Schwartz&Hearst

    2 <= len(str) <= 10
    len(tokens) <= 2
    re.search('[A-Za-z]', str)
    str[0].isalnum()

    and extra:
    if it matches ([A-Za-z]\. ?){2,}
    it is a good candidate.

    '''
    if not __REPRODUCE__ and re.match('([A-Za-z]\. ?){2,}', str.lstrip()):
        return True
    if len(str) < 2 or len(str) > 10:
        return False
    if len(str.split()) > 2:
        return False
    if not re.search('[A-Za-z]', str):
        return False
    if not str[0].isalnum():
        return False

    return True

def getdefinition(candidate, sentence):
    '''Takes a candidate and a sentence and returns the definition candidate.

    The definintion candidate is the set of tokens (in front of the candidate)
    that starts with a token starting with the first character of the candidate'''
    # Take the tokens in front of the candidate
    tokens = sentence[:candidate.start-2].lower().split()

    # the char that we are looking for
    key = candidate[0].lower()

    # Count the number of tokens that start with the same character as the candidate
    firstchars = [t[0] for t in tokens]

    definitionfreq = firstchars.count(key)
    candidatefreq = candidate.lower().count(key)

    # Look for the list of tokens in front of candidate that
    # have a sufficient number of tokens starting with key
    if candidatefreq <= definitionfreq:
        # we should at least have a good number of starts
        count=0
        start=0
        startindex=len(firstchars)-1
        while count < candidatefreq:
            if abs(start) > len(firstchars):
                raise ValueError('not found')

            start-=1
            # Look up key in the definition
            try:
                startindex = firstchars.index(key, len(firstchars)+start)
            except ValueError:
                pass

            # Count the number of keys in definition
            count = firstchars[startindex:].count(key)

        # We found enough keys in the definition so return the definition as a
        # definition candidate
        start = len(' '.join(tokens[:startindex]))
        stop = candidate.start-2
        str = sentence[start:stop]

        # Remove whitespace
        start = start + len(str) - len(str.lstrip())
        stop = stop - len(str) + len(str.rstrip())
        str = sentence[start:stop]

        return Candidate(start, stop, str)


    else:
        #print 'S', sentence
        #print >>sys.stderr, 'KEY', key
        #print >>sys.stderr, 'TOKENS', tokens
        #print >>sys.stderr, 'ABBREV', candidate
        raise ValueError('There are less keys in the tokens in front of candidate than there are in the candidate')



def definitionselection(definition, abbrev):
    '''Takes a definition candidate and an abbreviation candidate
    and returns True if the chars in the abbreviation occur in the definition

    Based on
    A simple algorithm for identifying abbreviation definitions in biomedical texts, Schwartz & Hearst'''

    if len(definition) < len(abbrev):
        raise ValueError('Abbreviation is longer than definition')

    if abbrev in definition.split():
        raise ValueError('Abbreviation is full word of definition')

    sindex = -1
    lindex = -1

    while 1:
        try:
            longchar = definition[lindex].lower()
        except IndexError:
            #print definition, '||',abbrev
            raise


        shortchar = abbrev[sindex].lower()

        if not shortchar.isalnum():
            sindex-=1

        if sindex == -1*len(abbrev):
            if shortchar == longchar:
                if lindex == -1*len(definition) or not definition[lindex-1].isalnum():
                    break
                else:
                    lindex-=1
            else:
                lindex-=1

                if lindex == -1*(len(definition)+1):
                    raise ValueError('definition of "%s" not found in "%s"' %(abbrev, definition))

        else:
            if shortchar == longchar:
                sindex -=1
                lindex -=1
            else:
                lindex -= 1

    definition =  definition[lindex:len(definition)]

    tokens = len(definition.split())
    length = len(abbrev)

    if tokens > min([length+5, length*2]):
        raise ValueError('did not meet min(|A|+5, |A|*2) constraint')

    # Do not return definitions that contain unbalanced parentheses
    if not __REPRODUCE__:
        #print 'ccc', abbrev, definition, definition.count('('), definition.count(')')
        if definition.count('(') != definition.count(')'):
            raise ValueError('Unbalanced parentheses not allowed in a definition')

    return definition

def get_abbreviations(sents):
    abbrs = {}
    for sentence in sents:
        try:
            for candidate in getcandidates(sentence):
                try:
                    definition = getdefinition(candidate, sentence)
                except ValueError, e:
                    continue
                else:
                    try:
                        definition = definitionselection(definition, candidate)
                    except ValueError, e:
                        continue
                    except IndexError, e:
                        continue
                    else:
                        definition_text = definition.replace('\n', ' ').replace('\r', '').strip()
                        abbrs[candidate] = definition_text
        except Exception, e:
            continue
    return abbrs
