#if !defined(INCLUDED_BITS_H)
#define INCLUDED_BITS_H

#include "base.h"

typedef struct {
	Bits64 *pointer;
	Index   index; /* index begins at 1 from `*pointer` */
} BitLocation;

PUBLIC BitLocation FindSetBit         (Bits64 *p, Bits64 *q);
PUBLIC BitLocation FindClearBit       (Bits64 *p, Bits64 *q);
PUBLIC BitLocation ReverseFindSetBit  (Bits64 *p, Bits64 *q);
PUBLIC BitLocation ReverseFindClearBit(Bits64 *p, Bits64 *q);

PUBLIC BitLocation FindSetBits         (Size n, Bits64 *p, Bits64 *q);
PUBLIC BitLocation FindClearBits       (Size n, Bits64 *p, Bits64 *q);
PUBLIC BitLocation ReverseFindSetBits  (Size n, Bits64 *p, Bits64 *q);
PUBLIC BitLocation ReverseFindClearBits(Size n, Bits64 *p, Bits64 *q);

PUBLIC void SetBits         (Size n, BitLocation location);
PUBLIC void ClearBits       (Size n, BitLocation location);
PUBLIC void ReverseSetBits  (Size n, BitLocation location);
PUBLIC void ReverseClearBits(Size n, BitLocation location);

/******************************************************************************/

#if defined(IMPLEMENT_BITS_H)

#include <immintrin.h>

#if defined(_WIN32)
#include <intrin.h>
#endif

#if defined(__clang__) || defined(__GNUC__)
#define _ffsll(x) __builtin_ffsll(x)
#else
inline int _ffsll(long long int x) {
	int r;
	if (!_BitScanForward64(x)) r = -1;
	return r + 1;
}
#endif

/* NOTE(Emhyr): "reverse" means to decrement the word pointer. it doesn't scan the bits reversely */

static inline BitLocation DoFindBit(Boolean reverse, Boolean toClear, Bits64 *p, Bits64 *q) {
	BitLocation result;
	__mmask8 m;
	__m512i v, cmp;
	
	cmp = _mm512_set1_epi64(toClear ? -1 : 0);
	if (reverse) p -= 7;
	for (;;) {
		if (p == q) break;

		v = _mm512_load_epi64(p);
		m = _mm512_cmpneq_epu64_mask(v, cmp);
		if (m) break;

		long long inc = sizeof(v) / sizeof(*p);
		if (reverse) inc = -inc;
		p += inc;
	}

	int zcnt;
	if (reverse) zcnt = 7 - _lzcnt_u32((unsigned)m << 24);
	else zcnt = _tzcnt_u32(m);
	result.pointer = p + zcnt;
	if (result.pointer != q) {
		Bits64 x = *result.pointer;
		if (toClear) x = ~x;
		result.index = _ffsll(x) - 1;
	}
	return result;
}

BitLocation FindSetBit         (Bits64 *p, Bits64 *q) { return DoFindBit(0, 0, p, q); }
BitLocation FindClearBit       (Bits64 *p, Bits64 *q) { return DoFindBit(0, 1, p, q); }
BitLocation ReverseFindSetBit  (Bits64 *p, Bits64 *q) { return DoFindBit(1, 0, p, q); }
BitLocation ReverseFindClearBit(Bits64 *p, Bits64 *q) { return DoFindBit(1, 1, p, q); }

#define WIDTHOF(x) (sizeof(x) * 8)

/* NOTE(Emhyr): if n <= half a word size, we can buffer-shift bits into a word to
compare with a bitmask. */

BitLocation DoFindBits(Boolean reverse, Boolean toClear, Size n, Bits64 *p, Bits64 *q) {
	if (n == 1) {
		if (reverse) {
			if (toClear) return ReverseFindClearBit(p, q);
			else return ReverseFindSetBit(p, q);
		} else {
			if (toClear) return FindClearBit(p, q);
			else return FindSetBit(p, q);
		}
	}
	
	BitLocation result;
	Size c, i, j, z;
	Bits64 w;
	
	result.pointer = p;
	result.index = 0;
	c = 0;
	for (; p != q; reverse ? --p : ++p) {
		w = *p;
		if (toClear) w = ~w;
	retry:
		result.pointer = p;
		i = _ffsll(w);
		if (!i) continue;
		result.index = i - 1;
		w |= ~(-1ll << (i - 1));
		j = _ffsll(~w);
		if (!j) j = WIDTHOF(*p) + 1;
		c += j - i;
		if (c >= n) break;
		w &= -1ll << (j - 1);
		if (!w) continue;
		if (j == WIDTHOF(*p) + 1) {
			do {
				w = *++p;
				if (!toClear) w = ~w;
				if (p == q) goto finished;
				z = _tzcnt_u64(w);
				c += z;
				if (c >= n) goto finished;
			} while (z == WIDTHOF(*p));
			w |= ~(-1ll << (z - 1));
			w = ~w;
		}
		c = 0;
		goto retry;
	}
finished:
	if (p == q) result = (BitLocation){0, 0};
	return result;
}

BitLocation FindSetBits         (Size n, Bits64 *p, Bits64 *q) { return DoFindBits(0, 0, n, p, q); }
BitLocation FindClearBits       (Size n, Bits64 *p, Bits64 *q) { return DoFindBits(0, 1, n, p, q); }
BitLocation ReverseFindSetBits  (Size n, Bits64 *p, Bits64 *q) { return DoFindBits(1, 0, n, p, q); }
BitLocation ReverseFindClearBits(Size n, Bits64 *p, Bits64 *q) { return DoFindBits(1, 1, n, p, q); }

static inline void DoSetBits(Boolean reverse, Boolean clear, Size n, BitLocation location) {
	Assert(n);
	Assert(location.pointer && location.index < WIDTHOF(*location.pointer));

	Bits64 *p, m;
	long long i, k;
	Size c;

	p = location.pointer;
	*p = 0;
	c = WIDTHOF(*location.pointer) - location.index;
	if (n < c) c = n;
	m = ~(c < 64 ? -1ll << c : 0) << location.index;
	if (clear) *p &= ~m;
	else *p |= m;
	if (reverse) --p;
	else ++p;
	n -= c;
	if (!n) return;
	c = WIDTHOF(*location.pointer);
	k = n / c;
	n -= c * k;
	m = clear ? 0 : -1ll;
	for (i = 0; i < k; ++i) {
		*p = m;
		if (reverse) --p;
		else ++p;
	}
	m = -1ll >> (c - n);
	if (clear) *p &= ~m;
	else *p |= m;
}

void SetBits         (Size n, BitLocation location) { return DoSetBits(0, 0, n, location); }
void ClearBits       (Size n, BitLocation location) { return DoSetBits(0, 1, n, location); }
void ReverseSetBits  (Size n, BitLocation location) { return DoSetBits(1, 0, n, location); }
void ReverseClearBits(Size n, BitLocation location) { return DoSetBits(1, 1, n, location); }

#endif

#endif
