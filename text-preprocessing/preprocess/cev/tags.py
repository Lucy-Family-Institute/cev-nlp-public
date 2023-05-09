from typeguard import typechecked
import re

separators = ["-", "–"]


def parse_time(time):
    if ":" in time:
        sep = ":"
    elif "." in time:
        sep = "."
    else:
        return f"CouldNotParseError: Invalid time: {time}"
    items = time.split(sep)[::-1]
    seconds = 0
    for i, item in enumerate(items):
        seconds += int(item) * (60 ** i)
    return seconds


def obtain_times(tag, tag_name, num_times):
    """
    Given a tag, and a tag_name acting as the tag type, this function
    finds the number of seconds from the start of the interview until
    the tag was marked. If `num_times` is 1, only one time will be found.
    If `num_times` is 2, two times will be found.

    Parameters
    ----------
    tag : str
        A complete tag
    tag_name : str
        The name specifying the type of the tag
    num_times : int
        Number of times to be retrieved (must be 1 or 2).

    Returns
    -------
    str, int or tuple(int)
        If any parsing error is found, the function will return a string
        containing a CouldNotParseError message. If the function is successful,
        either an int (number of seconds) for a single time, or a tuple of ints
        for two times will be returned.
    """
    if num_times not in [1, 2]:
        raise ValueError(
            "Invalid number of times to retrieve/parse. You are trying "
            f" to obtain {num_times}, but you must select either 1 or 2"
        )
    try:
        times = tag.split(tag_name)[1].strip()
    except IndexError as e:
        # This happens because the tag is not, e.g. INAD: xx:yy:zz, but INAD xx:yy:zz
        times = tag.split(tag_name.replace(":", ""))[1].strip()
    start = None
    end = None
    if num_times == 2:
        for sep in separators:
            if sep in times:
                times_list = times.split(sep)
                if len(times_list) != 2:
                    return f"CouldNotParseError: many values to unpack in {times}"
                else:
                    start, end = times_list
                break
        if start is None:
            return f"CouldNotParseError: none of the separators {separators} are in the tag"
    elif num_times == 1:
        start = times
    try:
        start_seconds = parse_time(start)
    except:
        return f"CouldNotParseError: could not get the seconds corresponding to the time {start}"
    try:
        if num_times == 2:
            end_seconds = parse_time(end)
            return start_seconds, end_seconds
        elif num_times == 1:
            return start_seconds
    except:
        return f"CouldNotParseError: could not get the seconds corresponding to the time {end}"


def process_any_time_tag(tag, tag_name, num_times):
    """
    This function helps to process a time tag (one-time tag or two-times tag).
    The description of the parameters is the same as in `obtain_times`.
    """
    result = {"type": tag_name, "original": tag}
    times = obtain_times(tag, tag_name + ":", num_times)
    if isinstance(times, str):
        result["info"] = times
    elif isinstance(times, int):
        result["info"] = {"time_in_seconds": times}
    else:
        result["info"] = {
            "start_time_in_seconds": times[0],
            "end_time_in_seconds": times[1],
        }
    return result


def process_one_or_two_times_tag(tag, tag_name):
    """
    Process a time tag that can contain either one time or two times.
    """
    result = process_any_time_tag(tag, tag_name, 1)
    if isinstance(result, str):
        result = process_any_time_tag(tag, tag_name, 2)
    return result


@typechecked
def parse_tag(tag: str, verbose: bool = False):
    """
    Once a tag has been identified, this function determines the
    tag type, as well as the information associated with the tag.
    For a definition of the information that can be associeated
    with each tag, check
    https://gitlab.com/nlp-comision/survey-analysis/-/issues/31#note_485897258

    Parameters
    ----------
    tag : str
        A tag, which is found on transcriptions within brackets.
    verbose : bool
        If True, everything that cannot be parsed will be printed
        for further investigation.

    Returns
    -------
    dict
        A dictionary with the following keys:
        - "type": the tag type, e.g. PAUSA, risas, llanto, etc.
        - "info": A dictionary that contains more info about the tag
            that is being processed. An example of further information
            can be the processed times.
        - "original": the original tag that was received.

        An important remark is that if some expected parsing of the "info"
        could not be done, a message is saved in "info" with a leading string
        saying "CouldNotParseError: ..."
    """
    if "INTERRUP" in tag:
        result = {"type": "INTERRUP", "info": None, "original": tag}
    elif "CONT" in tag:
        result = {"type": "CONT", "info": None, "original": tag}
    elif "INAD" in tag:
        result = process_any_time_tag(tag, "INAD", 1)
    elif "DUD" in tag:
        result = process_one_or_two_times_tag(tag, "DUD")
    elif "CORTE" in tag:
        result = process_any_time_tag(tag, "CORTE", 1)
    elif "PAUSA" in tag:
        result = process_any_time_tag(tag, "PAUSA", 2)
    elif "Datos sensibles" in tag:
        result = process_any_time_tag(tag, "Datos sensibles", 2)
    elif "Datos personales" in tag:
        result = process_any_time_tag(tag, "Datos personales", 2)
    elif "INC:" in tag:
        result = {"type": "INC", "original": tag}
        content = tag.split("INC:")[1].strip()
        result["info"] = {"text": content}
    elif "risa" in tag:
        result = {"type": "risas", "original": tag, "info": None}
    elif "llanto" in tag:
        result = {"type": "llanto", "original": tag, "info": None}
    else:
        result = {
            "type": None,
            "original": tag,
            "info": f"CouldNotParseError: tag type not detected for tag '{tag}'",
        }
    if verbose and "CouldNotParseError" in result["info"]:
        print(result["info"])

    return result

tag_regex_pattern = re.compile(r"\[(.*?)\]")

def identify_tags(text):
    """
    Function that identifies all tags in an interview

    Returns
    -------
    tuple
        A tuple containing the character span and the text from the text
    """

    tag_texts = []
    for tag in tag_regex_pattern.finditer(text):
        start, end = tag.span()
        tag_texts.append(text[start+1:end-1]) # avoid brackets
    return tag_texts


def delete_tags(text):
    """
    Function that modifies a text, removing all tags

    Returns
    -------
    str
        The string of text without tags.
    """
    without_tags = re.sub(tag_regex_pattern, "", text)
    return re.sub(r"\s+", " ", without_tags).strip()
