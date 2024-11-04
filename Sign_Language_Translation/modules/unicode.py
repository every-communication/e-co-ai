# 자음모음 결합용
import itertools

INITIAL = 0x001
MEDIAL = 0x010
FINAL = 0x100
CHAR_LISTS = {
    INITIAL: list(map(chr, [
        0x3131, 0x3132, 0x3134, 0x3137, 0x3138, 0x3139,
        0x3141, 0x3142, 0x3143, 0x3145, 0x3146, 0x3147,
        0x3148, 0x3149, 0x314a, 0x314b, 0x314c, 0x314d,
        0x314e
    ])),
    MEDIAL: list(map(chr, [
        0x314f, 0x3150, 0x3151, 0x3152, 0x3153, 0x3154,
        0x3155, 0x3156, 0x3157, 0x3158, 0x3159, 0x315a,
        0x315b, 0x315c, 0x315d, 0x315e, 0x315f, 0x3160,
        0x3161, 0x3162, 0x3163
    ])),
    FINAL: list(map(chr, [
        0x3131, 0x3132, 0x3133, 0x3134, 0x3135, 0x3136,
        0x3137, 0x3139, 0x313a, 0x313b, 0x313c, 0x313d,
        0x313e, 0x313f, 0x3140, 0x3141, 0x3142, 0x3144,
        0x3145, 0x3146, 0x3147, 0x3148, 0x314a, 0x314b,
        0x314c, 0x314d, 0x314e
    ]))
}
CHAR_INITIALS = CHAR_LISTS[INITIAL]
CHAR_MEDIALS = CHAR_LISTS[MEDIAL]
CHAR_FINALS = CHAR_LISTS[FINAL]
CHAR_SETS = {k: set(v) for k, v in CHAR_LISTS.items()}
CHARSET = set(itertools.chain(*CHAR_SETS.values()))
CHAR_INDICES = {k: {c: i for i, c in enumerate(v)}
                for k, v in CHAR_LISTS.items()}


def is_hangul_syllable(c):
    return 0xac00 <= ord(c) <= 0xd7a3  # Hangul Syllables


def is_hangul_jamo(c):
    return 0x1100 <= ord(c) <= 0x11ff  # Hangul Jamo


def is_hangul_compat_jamo(c):
    return 0x3130 <= ord(c) <= 0x318f  # Hangul Compatibility Jamo


def is_hangul_jamo_exta(c):
    return 0xa960 <= ord(c) <= 0xa97f  # Hangul Jamo Extended-A


def is_hangul_jamo_extb(c):
    return 0xd7b0 <= ord(c) <= 0xd7ff  # Hangul Jamo Extended-B


def is_hangul(c):
    return (is_hangul_syllable(c) or
            is_hangul_jamo(c) or
            is_hangul_compat_jamo(c) or
            is_hangul_jamo_exta(c) or
            is_hangul_jamo_extb(c))


def is_supported_hangul(c):
    return is_hangul_syllable(c) or is_hangul_compat_jamo(c)


def check_hangul(c, jamo_only=False):
    if not ((jamo_only or is_hangul_compat_jamo(c)) or is_supported_hangul(c)):
        raise ValueError(f"'{c}' is not a supported hangul character. "
                         f"'Hangul Syllables' (0xac00 ~ 0xd7a3) and "
                         f"'Hangul Compatibility Jamos' (0x3130 ~ 0x318f) are "
                         f"supported at the moment.")


def get_jamo_type(c):
    check_hangul(c)
    assert is_hangul_compat_jamo(c), f"not a jamo: {ord(c):x}"
    return sum(t for t, s in CHAR_SETS.items() if c in s)


def split_syllable_char(c):
    """
    Splits a given korean syllable into its components. Each component is
    represented by Unicode in 'Hangul Compatibility Jamo' range.

    Arguments:
        c: A Korean character.

    Returns:
        A triple (initial, medial, final) of Hangul Compatibility Jamos.
        If no jamo corresponds to a position, `None` is returned there.

    Example:
        >>> split_syllable_char("안")
        ("ㅇ", "ㅏ", "ㄴ")
        >>> split_syllable_char("고")
        ("ㄱ", "ㅗ", None)
        >>> split_syllable_char("ㅗ")
        (None, "ㅗ", None)
        >>> split_syllable_char("ㅇ")
        ("ㅇ", None, None)
    """
    check_hangul(c)
    if len(c) != 1:
        raise ValueError("Input string must have exactly one character.")

    init, med, final = None, None, None
    if is_hangul_syllable(c):
        offset = ord(c) - 0xac00
        x = (offset - offset % 28) // 28
        init, med, final = x // 21, x % 21, offset % 28
        if not final:
            final = None
        else:
            final -= 1
    else:
        pos = get_jamo_type(c)
        if pos & INITIAL == INITIAL:
            pos = INITIAL
        elif pos & MEDIAL == MEDIAL:
            pos = MEDIAL
        elif pos & FINAL == FINAL:
            pos = FINAL
        idx = CHAR_INDICES[pos][c]
        if pos == INITIAL:
            init = idx
        elif pos == MEDIAL:
            med = idx
        else:
            final = idx
    return tuple(CHAR_LISTS[pos][idx] if idx is not None else None
                 for pos, idx in
                 zip([INITIAL, MEDIAL, FINAL], [init, med, final]))

# 어절(단어) -> 음소(자모)
def split_syllables(s, ignore_err=True, pad=None):
    """
    Performs syllable-split on a string.

    Arguments:
        s (str): A string (possibly mixed with non-Hangul characters).
        ignore_err (bool): If set False, it ensures that all characters in
            the string are Hangul-splittable and throws a ValueError otherwise.
            (default: True)
        pad (str): Pad empty jamo positions (initial, medial, or final) with
            `pad` character. This is useful for cases where fixed-length
            strings are needed. (default: None)

    Returns:
        Hangul-split string

    Example:
        >>> split_syllables("안녕하세요")
        "ㅇㅏㄴㄴㅕㅇㅎㅏㅅㅔㅇㅛ"
        >>> split_syllables("안녕하세요~~", ignore_err=False)
        ValueError: encountered an unsupported character: ~ (0x7e)
        >>> split_syllables("안녕하세요ㅛ", pad="x")
        'ㅇㅏㄴㄴㅕㅇㅎㅏxㅅㅔxㅇㅛxxㅛx'
    """

    def try_split(c):
        try:
            return split_syllable_char(c)
        except ValueError:
            if ignore_err:
                return (c,)
            raise ValueError(f"encountered an unsupported character: "
                             f"{c} (0x{ord(c):x})")

    s = map(try_split, s)
    if pad is not None:
        tuples = map(lambda x: tuple(pad if y is None else y for y in x), s)
    else:
        tuples = map(lambda x: filter(None, x), s)
    return "".join(itertools.chain(*tuples))

def comb_final_sub(c1, c2):
    if c1 == 'ㄱ' and c2 == 'ㅅ':
        return 'ㄳ'
    if c1 == 'ㄴ':
        if c2 == 'ㅈ':
            return 'ㄵ'
        if c2 == 'ㅎ':
            return 'ㄶ'
    if c1 == 'ㄹ':
        if c2 == 'ㄱ':
            return 'ㄺ'
        if c2 == 'ㅁ':
            return 'ㄻ'
        if c2 == 'ㅂ':
            return 'ㄼ'
        if c2 == 'ㅅ':
            return 'ㄽ'
        if c2 == 'ㅌ':
            return 'ㄾ'
        if c2 == 'ㅍ':
            return 'ㄿ'
        if c2 == 'ㅎ':
            return 'ㅀ'
    if c1 == 'ㅂ' and c2 == 'ㅅ':
        return 'ㅄ'
    return None

# 음소(자모) -> 음절
def join_jamos_char(init, med, final=None, sub=None):
    """
    Combines jamos into a single syllable.

    Arguments:
        init (str): Initial jao.
        med (str): Medial jamo.
        final (str): Final jamo. If not supplied, the final syllable is made
            without the final. (default: None)

    Returns:
        A Korean syllable.
    """
    if sub is not None:
        combined_final = comb_final_sub(final, sub)
        if combined_final:
            final = combined_final
            
    chars = (init, med, final)
    for c in filter(None, chars):
        check_hangul(c, jamo_only=True)

    idx = tuple(CHAR_INDICES[pos][c] if c is not None else c
                for pos, c in zip((INITIAL, MEDIAL, FINAL), chars))
    init_idx, med_idx, final_idx = idx
    # final index must be shifted once as
    # final index with 0 points to syllables without final
    final_idx = 0 if final_idx is None else final_idx + 1
    return chr(0xac00 + 28 * 21 * init_idx + 28 * med_idx + final_idx)

# 음소(자모) -> 어절(단어)
def join_jamos(s, ignore_err=True):
    """
    Combines a sequence of jamos to produce a sequence of syllables.

    Arguments:
        s (str): A string (possible mixed with non-jamo characters).
        ignore_err (bool): If set False, it will ensure that all characters
            will be consumed for the making of syllables. It will throw a
            ValueError when it fails to do so. (default: True)

    Returns:
        A string

    Example:
        >>> join_jamos("ㅇㅏㄴㄴㅕㅇㅎㅏㅅㅔㅇㅛ")
        "안녕하세요"
        >>> join_jamos("ㅇㅏㄴㄴㄴㅕㅇㅎㅏㅅㅔㅇㅛ")
        "안ㄴ녕하세요"
        >>> join_jamos()
    """
    last_t = 0
    queue = []
    new_string = ""

    def flush(n=0):
        new_queue = []
        while len(queue) > n:
            new_queue.append(queue.pop())
        if len(new_queue) == 1:
            if not ignore_err:
                raise ValueError(f"invalid jamo character: {new_queue[0]}")
            result = new_queue[0]
        elif len(new_queue) >= 2:
            try:
                result = join_jamos_char(*new_queue)
            except (ValueError, KeyError):
                # Invalid jamo combination
                if not ignore_err:
                    raise ValueError(f"invalid jamo characters: {new_queue}")
                result = "".join(new_queue)
        else:
            result = None
        return result

    # Check Medial Index
    medial_index = []
    for i in range(len(s)):
        if s[i] in CHARSET:
            t = get_jamo_type(s[i])
        else:
            t = 0

        if t == MEDIAL:
            medial_index.append(i)
    
    last_index = -1
    for i in range(len(medial_index)):
        if medial_index[i] == last_index + 1: # 모음으로 시작
            new_string += s[medial_index[i]]
            last_index = medial_index[i]
        else:
            # 이전까지 값 넣음
            for t in range(last_index + 1, medial_index[i] - 1):
                new_string += s[t]
                last_index = t
            
            # 5개의 글자의 인덱스
            twi = []
            for t in range(medial_index[i] - 1, medial_index[i] + 4):
                if t < len(s):
                    twi.append(t)
            
            if s[twi[0]] in CHARSET:
                t = get_jamo_type(s[twi[0]])
            else:
                t = 0
            if t & INITIAL != INITIAL:
                new_string += s[twi[0]]
                last_index = twi[0]
                continue
            
            queue.insert(0, s[twi[0]])
            
            insert_into_new_string = False
            # 모음 위치 기반 처리
            for j in range(len(twi)):
                insert_into_new_string = False
                if s[twi[j]] in CHARSET:
                    t = get_jamo_type(s[twi[j]])
                else:
                    t = 0
                
                if t == MEDIAL:
                    if j == 1: # 현재 음절
                        queue.insert(0, s[twi[1]])
                        continue
                    
                    if j == 2 or j == 3: # 자 모 모 or 자 모 자 모
                        new_string += flush()
                        last_index = twi[1]
                        insert_into_new_string = True
                        break
                    
                    if j == 4: # 자 모 자 자 모
                        if s[twi[2]] in CHARSET:
                            t_next = get_jamo_type(s[twi[2]])
                        else:
                            t_next = 0

                        if t_next & FINAL == FINAL:
                            queue.insert(0, s[twi[2]])
                            new_string += flush()
                        else:
                            new_string += flush() + s[twi[2]]
                        last_index = twi[2]
                        insert_into_new_string = True
                        break
            
            if insert_into_new_string:
                continue
                
            # 자 모 자 자 자 and 받침 안 넣은 상태
            if len(twi) == 2:
                new_string += flush()
                last_index = twi[1]
                continue
            
            if len(twi) == 3:
                if s[twi[2]] in CHARSET:
                    t_next = get_jamo_type(s[twi[2]])
                else:
                    t_next = 0

                if t_next & FINAL == FINAL:
                    queue.insert(0, s[twi[2]])
                    new_string += flush()
                else:
                    new_string += flush() + s[twi[2]]
                last_index = twi[2]
                continue

            if comb_final_sub(s[twi[2]], s[twi[3]]):
                queue.insert(0, s[twi[2]])
                queue.insert(0, s[twi[3]])
                new_string += flush()
                last_index = twi[3]
                continue
            else:
                if s[twi[2]] in CHARSET:
                    t_next = get_jamo_type(s[twi[2]])
                else:
                    t_next = 0
                if t_next & FINAL == FINAL:
                    queue.insert(0, s[twi[2]])
                    new_string += flush() + s[twi[3]]
                else:
                    new_string += flush() + s[twi[2]] + s[twi[3]]
                last_index = twi[3]
                continue
            

    for remain_char_index in range(last_index + 1, len(s)):
        new_string += s[remain_char_index]

    return new_string

            
    """         
    for c in s:
        if c not in CHARSET:
            if queue:
                new_c = flush() + c
            else:
                new_c = c
            last_t = 0
        else:
            t = get_jamo_type(c)
            new_c = None
            if t & FINAL == FINAL:
                if not (last_t == MEDIAL):
                    new_c = flush()
            elif t == INITIAL:
                new_c = flush()
            elif t == MEDIAL:
                if last_t & INITIAL == INITIAL:
                    new_c = flush(1)
                else:
                    new_c = flush()
            last_t = t
            queue.insert(0, c)
        if new_c:
            new_string += new_c
    if queue:
        new_string += flush()
    return new_string
    
    """


dc_befor = ['ㄱ', 'ㄷ', 'ㅂ', 'ㅅ', 'ㅈ']
dc_after = ['ㄲ', 'ㄸ', 'ㅃ', 'ㅆ', 'ㅉ']
def process_word(sentence, c):
    #TODO 문제 발생 시 중복 안되게 막으면 됨
    if len(sentence) > 0:
        if sentence[-1] == c:
            return sentence, join_jamos(sentence)
    
    if c == 'Space':
        sentence = sentence + " "

    elif c == 'Back':
        if len(sentence) != 0:
            sentence = sentence[:-1]
    
    elif c == 'Double':
        for i in range(len(dc_befor)):
            if sentence[-1] == dc_befor[i]:
                sentence = sentence[:-1]
                sentence += dc_after[i]

    elif c == 'Clear':
        sentence = ""

    else:
        sentence += c
    
    result = join_jamos(sentence)
    return sentence, result


if __name__=="__main__":
    s = ""
    while True:
        w = input("입력:")
        if w == 'q':
            break

        s, res = process_word(s, w)

        print("문장 : ", s)
        print("결과 : ", res)

    temp = ['간', '다', '갊', '가나']
    for word in temp:
        print(split_syllables(word))